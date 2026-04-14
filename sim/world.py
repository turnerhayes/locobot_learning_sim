"""
SimWorld: 2D physics-based simulation of the RecycleBot environment.

This replaces the real LocoBot + ROS services with a Pymunk-backed 2D world.
The robot operates on a ground plane with:
  - Two rooms connected by a doorway
  - A ball (graspable object)
  - A bin (target container) in room 2
  - Novelty objects: curtain (hinged obstacle), box (pushable rigid body)

Coordinate system:
  - Room 1: x in [0, 4], y in [0, 3]
  - Doorway: centered at x=4, y=1.5, width ~0.8m
  - Room 2: x in [4, 8], y in [0, 3]
  - Robot starts at (1.0, 1.5) facing right (heading=0)
  - Ball starts at (1.5, 1.5) in room 1
  - Bin at (6.5, 1.5) in room 2

Units are meters. The robot is a differential-drive disc (radius ~0.17m, matching LocoBot).
"""

import math
import numpy as np
import pymunk
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# Constants
# ============================================================

ROBOT_RADIUS = 0.17  # LocoBot base radius in meters
ROBOT_MASS = 5.0
BALL_RADIUS = 0.05
BALL_MASS = 0.1
BOX_SIZE = (0.3, 0.3)  # width, height of the box obstacle
BOX_MASS = 2.0
CURTAIN_WIDTH = 0.05
CURTAIN_LENGTH = 0.6  # how far the curtain extends from the wall

WALL_THICKNESS = 0.1

# Room geometry
ROOM_WIDTH = 4.0
ROOM_HEIGHT = 3.0
DOORWAY_Y_CENTER = 1.5
DOORWAY_HALF_WIDTH = 0.4  # doorway opening is 0.8m wide

# Collision categories (bitmask)
CAT_WALL = 0b0001
CAT_ROBOT = 0b0010
CAT_OBJECT = 0b0100  # ball, box
CAT_CURTAIN = 0b1000

# Physics timestep
DT = 1.0 / 60.0  # 60 Hz physics
SUBSTEPS = 10  # substeps per action step


class NoveltyType(Enum):
    NONE = "none"
    CURTAIN = "curtain"
    BOX = "box"
    BALL_OBSTACLE = "ball_obstacle"


@dataclass
class WorldConfig:
    """Configuration for the simulation world."""
    # Room layout
    room_width: float = ROOM_WIDTH
    room_height: float = ROOM_HEIGHT
    doorway_y_center: float = DOORWAY_Y_CENTER
    doorway_half_width: float = DOORWAY_HALF_WIDTH

    # Initial positions
    robot_start: Tuple[float, float] = (1.0, 1.5)
    robot_start_heading: float = 0.0  # facing right (toward doorway)
    ball_start: Tuple[float, float] = (1.5, 1.5)
    bin_position: Tuple[float, float] = (6.5, 1.5)

    # Novelty config
    novelty: NoveltyType = NoveltyType.NONE
    box_position: Tuple[float, float] = (5.8, 1.5)  # blocks path to bin
    curtain_anchor: Tuple[float, float] = (4.0, DOORWAY_Y_CENTER + DOORWAY_HALF_WIDTH)
    obstacle_ball_position: Tuple[float, float] = (5.8, 1.5)  # ball blocking bin
    obstacle_ball_radius: float = 0.1  # larger than task ball
    obstacle_ball_mass: float = 0.3

    # Physics
    linear_velocity: float = 0.5  # m/s for move_forward
    angular_velocity: float = 0.35  # rad/s for turn actions (matching real robot defaults)
    action_duration: float = 0.5  # seconds per primitive action step

    # Friction
    floor_friction: float = 0.8
    box_friction: float = 0.6
    curtain_spring_stiffness: float = 5.0
    curtain_spring_damping: float = 2.0


class SimWorld:
    """
    2D physics world for the RecycleBot environment.

    Provides:
      - Physics stepping with Pymunk
      - Robot control (move_forward, turn_left, turn_right)
      - Symbolic state queries (at, facing, hold, contain)
      - Sub-symbolic state (local occupancy grid, relative poses)
      - Novelty injection (curtain, box, ball drift)
    """

    def __init__(self, config: Optional[WorldConfig] = None):
        self.config = config or WorldConfig()
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # top-down 2D, no gravity
        self.space.damping = 0.3  # global damping to slow things down

        # State tracking
        self.robot_body: Optional[pymunk.Body] = None
        self.robot_shape: Optional[pymunk.Shape] = None
        self.ball_body: Optional[pymunk.Body] = None
        self.ball_shape: Optional[pymunk.Shape] = None
        self.box_body: Optional[pymunk.Body] = None
        self.box_shape: Optional[pymunk.Shape] = None
        self.obstacle_ball_body: Optional[pymunk.Body] = None
        self.obstacle_ball_shape: Optional[pymunk.Shape] = None
        self.curtain_body: Optional[pymunk.Body] = None
        self.curtain_shape: Optional[pymunk.Shape] = None
        self.curtain_joint: Optional[pymunk.Constraint] = None
        self.curtain_spring: Optional[pymunk.Constraint] = None

        self.walls: List[pymunk.Shape] = []

        # Symbolic state
        self._robot_holding: Optional[str] = None  # None or "ball_1"
        self._ball_in_bin: bool = False
        self._ball_picked: bool = False  # ball has been removed from world (held by robot)

        # Room boundaries (as shapely-compatible polygon coords)
        self.room_boundaries = {
            "room_1": [(0, 0), (self.config.room_width, 0),
                       (self.config.room_width, self.config.room_height),
                       (0, self.config.room_height)],
            "room_2": [(self.config.room_width, 0),
                       (2 * self.config.room_width, 0),
                       (2 * self.config.room_width, self.config.room_height),
                       (self.config.room_width, self.config.room_height)],
        }

        # Facing boundaries: approximate regions where robot is considered "facing" an object
        # These are computed dynamically based on object positions and robot heading
        self.facing_distance_threshold = 0.6  # must be within this distance
        self.facing_angle_threshold = 0.5  # radians, angular tolerance

        self._build_world()

    def _build_world(self):
        """Construct the full environment: walls, robot, objects, novelties."""
        self._build_walls()
        self._build_robot()
        self._build_ball()
        self._build_novelties()

    def _build_walls(self):
        """Build the wall segments for two rooms with a doorway between them."""
        rw = self.config.room_width
        rh = self.config.room_height
        dy = self.config.doorway_y_center
        dhw = self.config.doorway_half_width
        t = WALL_THICKNESS

        static_body = self.space.static_body

        wall_segments = [
            # Room 1 outer walls
            ((0, 0), (rw, 0)),           # bottom wall room 1
            ((0, 0), (0, rh)),           # left wall room 1
            ((0, rh), (rw, rh)),         # top wall room 1

            # Room 2 outer walls
            ((rw, 0), (2*rw, 0)),        # bottom wall room 2
            ((2*rw, 0), (2*rw, rh)),     # right wall room 2
            ((rw, rh), (2*rw, rh)),      # top wall room 2

            # Dividing wall with doorway gap
            # Bottom section of dividing wall (below doorway)
            ((rw, 0), (rw, dy - dhw)),
            # Top section of dividing wall (above doorway)
            ((rw, dy + dhw), (rw, rh)),
        ]

        for (x1, y1), (x2, y2) in wall_segments:
            segment = pymunk.Segment(static_body, (x1, y1), (x2, y2), t / 2)
            segment.elasticity = 0.2
            segment.friction = 1.0
            segment.filter = pymunk.ShapeFilter(categories=CAT_WALL)
            self.space.add(segment)
            self.walls.append(segment)

    def _build_robot(self):
        """Create the robot as a kinematic body (we control velocity directly)."""
        mass = ROBOT_MASS
        moment = pymunk.moment_for_circle(mass, 0, ROBOT_RADIUS)
        self.robot_body = pymunk.Body(mass, moment)
        self.robot_body.position = self.config.robot_start
        self.robot_body.angle = self.config.robot_start_heading

        self.robot_shape = pymunk.Circle(self.robot_body, ROBOT_RADIUS)
        self.robot_shape.elasticity = 0.1
        self.robot_shape.friction = self.config.floor_friction
        self.robot_shape.filter = pymunk.ShapeFilter(
            categories=CAT_ROBOT,
            mask=CAT_WALL | CAT_OBJECT | CAT_CURTAIN
        )

        self.space.add(self.robot_body, self.robot_shape)

    def _build_ball(self):
        """Create the ball object."""
        mass = BALL_MASS
        moment = pymunk.moment_for_circle(mass, 0, BALL_RADIUS)
        self.ball_body = pymunk.Body(mass, moment)
        self.ball_body.position = self.config.ball_start

        self.ball_shape = pymunk.Circle(self.ball_body, BALL_RADIUS)
        self.ball_shape.elasticity = 0.5
        self.ball_shape.friction = 0.5
        self.ball_shape.filter = pymunk.ShapeFilter(
            categories=CAT_OBJECT,
            mask=CAT_WALL | CAT_ROBOT | CAT_OBJECT
        )

        self.space.add(self.ball_body, self.ball_shape)

    def _build_novelties(self):
        """Build novelty objects based on config."""
        if self.config.novelty == NoveltyType.BOX:
            self._build_box()
        elif self.config.novelty == NoveltyType.CURTAIN:
            self._build_curtain()
        elif self.config.novelty == NoveltyType.BALL_OBSTACLE:
            self._build_obstacle_ball()

    def _build_box(self):
        """Create a pushable box obstacle."""
        w, h = BOX_SIZE
        mass = BOX_MASS
        moment = pymunk.moment_for_box(mass, (w, h))
        self.box_body = pymunk.Body(mass, moment)
        self.box_body.position = self.config.box_position

        self.box_shape = pymunk.Poly.create_box(self.box_body, (w, h))
        self.box_shape.elasticity = 0.1
        self.box_shape.friction = self.config.box_friction
        self.box_shape.filter = pymunk.ShapeFilter(
            categories=CAT_OBJECT,
            mask=CAT_WALL | CAT_ROBOT | CAT_OBJECT
        )

        self.space.add(self.box_body, self.box_shape)

    def _build_curtain(self):
        """
        Create a curtain as a hinged rigid body at the doorway.
        The curtain is attached to the top of the doorway frame with a pivot joint,
        and a damped rotary spring provides resistance.
        """
        rw = self.config.room_width
        anchor = self.config.curtain_anchor

        mass = 0.5
        length = CURTAIN_LENGTH
        w = CURTAIN_WIDTH
        moment = pymunk.moment_for_box(mass, (w, length))

        self.curtain_body = pymunk.Body(mass, moment)
        # Position the curtain hanging down from anchor point
        self.curtain_body.position = (anchor[0], anchor[1] - length / 2)

        self.curtain_shape = pymunk.Poly.create_box(self.curtain_body, (w, length))
        self.curtain_shape.elasticity = 0.1
        self.curtain_shape.friction = 0.3
        self.curtain_shape.filter = pymunk.ShapeFilter(
            categories=CAT_CURTAIN,
            mask=CAT_ROBOT
        )

        self.space.add(self.curtain_body, self.curtain_shape)

        # Pivot joint at the anchor point (top of curtain)
        self.curtain_joint = pymunk.PivotJoint(
            self.space.static_body, self.curtain_body, anchor
        )
        self.space.add(self.curtain_joint)

        # Spring that pulls curtain back to resting position (hanging straight down)
        self.curtain_spring = pymunk.DampedRotarySpring(
            self.space.static_body, self.curtain_body,
            rest_angle=0.0,
            stiffness=self.config.curtain_spring_stiffness,
            damping=self.config.curtain_spring_damping,
        )
        self.space.add(self.curtain_spring)

    def _build_obstacle_ball(self):
        """
        Create a ball obstacle blocking the bin/object.
        This is a separate physical ball (not the task ball_1) placed in front
        of the bin. The robot must push it aside to access the bin.
        Spherical shape means it rolls more easily than the box.
        """
        r = self.config.obstacle_ball_radius
        mass = self.config.obstacle_ball_mass
        moment = pymunk.moment_for_circle(mass, 0, r)
        self.obstacle_ball_body = pymunk.Body(mass, moment)
        self.obstacle_ball_body.position = self.config.obstacle_ball_position

        self.obstacle_ball_shape = pymunk.Circle(self.obstacle_ball_body, r)
        self.obstacle_ball_shape.elasticity = 0.4
        self.obstacle_ball_shape.friction = 0.3  # rolls more easily than box
        self.obstacle_ball_shape.filter = pymunk.ShapeFilter(
            categories=CAT_OBJECT,
            mask=CAT_WALL | CAT_ROBOT | CAT_OBJECT
        )

        self.space.add(self.obstacle_ball_body, self.obstacle_ball_shape)

    # ============================================================
    # Robot Actions
    # ============================================================

    def move_forward(self) -> None:
        """Apply forward velocity to the robot for one action step."""
        heading = self.robot_body.angle
        vx = self.config.linear_velocity * math.cos(heading)
        vy = self.config.linear_velocity * math.sin(heading)
        self.robot_body.velocity = (vx, vy)
        self._step_physics()
        self.robot_body.velocity = (0, 0)

    def turn_left(self) -> None:
        """Rotate the robot counter-clockwise for one action step."""
        delta = self.config.angular_velocity * self.config.action_duration
        self.robot_body.angle += delta
        self._step_physics_brief()

    def turn_right(self) -> None:
        """Rotate the robot clockwise for one action step."""
        delta = self.config.angular_velocity * self.config.action_duration
        self.robot_body.angle -= delta
        self._step_physics_brief()

    def approach_waypoint(self, target_name: str) -> bool:
        """
        Teleport/navigate the robot to a named waypoint.
        Returns True if successful (no obstacle blocking).
        This simulates the real robot's move_base navigation.
        """
        waypoints = self._get_waypoints()
        if target_name not in waypoints:
            return False

        target_pos, target_heading = waypoints[target_name]

        # Check if path is clear (simple line-of-sight check)
        if self._is_path_blocked(self.robot_body.position, target_pos):
            return False

        self.robot_body.position = target_pos
        self.robot_body.angle = target_heading
        self.robot_body.velocity = (0, 0)
        self.robot_body.angular_velocity = 0
        return True

    def pick_object(self) -> bool:
        """
        Attempt to pick up the ball. Succeeds if robot is close and facing it.
        """
        if self._robot_holding is not None:
            return False
        if self._ball_picked or self._ball_in_bin:
            return False

        ball_pos = self.ball_body.position
        robot_pos = self.robot_body.position
        dist = math.dist(robot_pos, ball_pos)

        if dist < ROBOT_RADIUS + BALL_RADIUS + 0.15 and self._is_facing_position(ball_pos):
            self._robot_holding = "ball_1"
            self._ball_picked = True
            # Remove ball from physics
            self.space.remove(self.ball_body, self.ball_shape)
            return True
        return False

    def place_object(self) -> bool:
        """
        Attempt to place the held object into the bin.
        Succeeds if robot is close to bin and facing it.
        """
        if self._robot_holding is None:
            return False

        bin_pos = self.config.bin_position
        robot_pos = self.robot_body.position
        dist = math.dist(robot_pos, bin_pos)

        if dist < ROBOT_RADIUS + 0.4 and self._is_facing_position(bin_pos):
            self._ball_in_bin = True
            self._robot_holding = None
            return True
        return False

    # ============================================================
    # Symbolic State Queries (replace ROS services)
    # ============================================================

    def query_at(self, obj: str, room: str) -> bool:
        """Check if an object is in a given room."""
        from shapely.geometry import Point, Polygon

        if obj == "doorway_1":
            return room in ("room_1", "room_2")

        if obj == "robot_1":
            pos = self.robot_body.position
        elif obj in ("ball_1", "can_1", "generic_object"):
            if self._robot_holding == "ball_1":
                # Ball is with robot
                pos = self.robot_body.position
            elif self._ball_in_bin:
                pos = self.config.bin_position
            elif self._ball_picked:
                pos = self.robot_body.position
            else:
                pos = self.ball_body.position
        elif obj == "bin_1":
            pos = self.config.bin_position
        else:
            return False

        boundary = self.room_boundaries.get(room)
        if boundary is None:
            return False

        point = Point(pos[0], pos[1])
        polygon = Polygon(boundary)
        return point.within(polygon)

    def query_facing(self, obj: str) -> bool:
        """Check if the robot is facing a given object."""
        if obj == "nothing":
            # True if not facing any specific object
            for check_obj in ["ball_1", "bin_1", "doorway_1", "generic_object", "table"]:
                if self._check_facing_object(check_obj):
                    return False
            return True

        return self._check_facing_object(obj)

    def query_hold(self, obj: str) -> bool:
        """Check if the robot is holding the given object."""
        if obj == "nothing":
            return self._robot_holding is None
        if obj in ("ball_1", "can_1"):
            return self._robot_holding == "ball_1"
        return False

    def query_contain(self, obj: str, container: str) -> bool:
        """Check if an object is contained in a container."""
        if container == "bin_1" and obj in ("ball_1", "can_1"):
            return self._ball_in_bin
        return False

    # ============================================================
    # Sub-symbolic State
    # ============================================================

    def get_local_occupancy_grid(self, size: int = 8, resolution: float = 0.1) -> np.ndarray:
        """
        Generate a local occupancy grid centered on the robot, aligned with robot heading.
        Returns an (size x size) grid with values:
          0.0 = free
          0.5 = curtain (semi-permeable, pushable)
          1.0 = occupied (wall, rigid obstacle)
        """
        grid = np.zeros((size, size), dtype=np.float32)
        robot_pos = self.robot_body.position
        robot_angle = self.robot_body.angle

        cos_a = math.cos(robot_angle)
        sin_a = math.sin(robot_angle)

        half = size / 2.0

        for gy in range(size):
            for gx in range(size):
                # Grid cell center in robot-local frame
                local_x = (gx - half + 0.5) * resolution
                local_y = (gy - half + 0.5) * resolution

                # Transform to world frame
                world_x = robot_pos[0] + local_x * cos_a - local_y * sin_a
                world_y = robot_pos[1] + local_x * sin_a + local_y * cos_a

                # Check curtain separately (0.5 encoding)
                curtain_query = self.space.point_query_nearest(
                    (world_x, world_y), 0.0,
                    pymunk.ShapeFilter(mask=CAT_CURTAIN)
                )
                if curtain_query is not None and curtain_query.distance <= resolution * 0.5:
                    grid[gy, gx] = 0.5
                    continue

                # Check walls and rigid objects (1.0 encoding)
                rigid_query = self.space.point_query_nearest(
                    (world_x, world_y), 0.0,
                    pymunk.ShapeFilter(mask=CAT_WALL | CAT_OBJECT)
                )
                if rigid_query is not None and rigid_query.distance <= resolution * 0.5:
                    grid[gy, gx] = 1.0

        return grid

    def get_relative_pose(self, target_name: str) -> Tuple[float, float]:
        """Get relative (x, y) position of a target w.r.t. the robot."""
        waypoints = self._get_waypoints()
        if target_name in waypoints:
            target_pos = waypoints[target_name][0]
        else:
            return (0.0, 0.0)

        robot_pos = self.robot_body.position
        rel_x = target_pos[0] - robot_pos[0]
        rel_y = target_pos[1] - robot_pos[1]
        return (rel_x, rel_y)

    def get_robot_position(self) -> Tuple[float, float]:
        """Get robot position in world frame."""
        return tuple(self.robot_body.position)

    def get_robot_heading(self) -> float:
        """Get robot heading in radians."""
        return self.robot_body.angle

    def is_forward_obstructed(self) -> bool:
        """Check if moving forward would result in a collision."""
        heading = self.robot_body.angle
        probe_dist = ROBOT_RADIUS + 0.05
        probe_x = self.robot_body.position[0] + probe_dist * math.cos(heading)
        probe_y = self.robot_body.position[1] + probe_dist * math.sin(heading)

        query = self.space.point_query_nearest(
            (probe_x, probe_y), ROBOT_RADIUS * 0.5,
            pymunk.ShapeFilter(mask=CAT_WALL | CAT_OBJECT)
        )
        return query is not None and query.distance < 0.01

    # ============================================================
    # Novelty-specific behavior
    # ============================================================

    # ============================================================
    # Reset
    # ============================================================

    def reset(self):
        """Reset the world to initial state."""
        # Remove everything and rebuild
        for shape in list(self.space.shapes):
            self.space.remove(shape)
        for body in list(self.space.bodies):
            self.space.remove(body)
        for constraint in list(self.space.constraints):
            self.space.remove(constraint)

        self.walls.clear()
        self._robot_holding = None
        self._ball_in_bin = False
        self._ball_picked = False

        self._build_world()

    # ============================================================
    # Internal Helpers
    # ============================================================

    def _step_physics(self):
        """Step physics for one action duration."""
        steps = int(self.config.action_duration / DT)
        for _ in range(steps):
            self.space.step(DT)

    def _step_physics_brief(self):
        """Brief physics step for rotations (just settle contacts)."""
        for _ in range(3):
            self.space.step(DT)

    def _get_waypoints(self) -> Dict[str, Tuple[Tuple[float, float], float]]:
        """
        Named waypoints mapping to (position, heading).
        Mirrors the real robot's nav_goals.
        """
        rw = self.config.room_width
        dy = self.config.doorway_y_center

        return {
            "generic_object": (
                (self.config.ball_start[0] - 0.3, self.config.ball_start[1]),
                0.0,  # face right toward ball
            ),
            "atdoor": ((rw - 0.3, dy), 0.0),  # in room 1, facing doorway
            "postdoor": ((rw + 0.3, dy), 0.0),  # just past doorway, in room 2
            "bin": (
                (self.config.bin_position[0] - 0.4, self.config.bin_position[1]),
                0.0,  # facing right toward bin
            ),
            "table": ((1.5, 0.3), math.pi / 2),  # placeholder
            "home": ((0.5, 1.5), 0.0),
        }

    def _is_facing_position(self, target_pos) -> bool:
        """Check if robot is roughly facing a given world position."""
        robot_pos = self.robot_body.position
        heading = self.robot_body.angle

        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        angle_to_target = math.atan2(dy, dx)

        angle_diff = abs((heading - angle_to_target + math.pi) % (2 * math.pi) - math.pi)
        return angle_diff < self.facing_angle_threshold

    def _check_facing_object(self, obj: str) -> bool:
        """Check if robot is facing a specific named object.
        For bin_1, also checks that the path is not obstructed (matching
        real robot's bin_visibility_checker).
        """
        pos = self._get_object_position(obj)
        if pos is None:
            return False

        robot_pos = self.robot_body.position
        dist = math.dist(robot_pos, pos)

        # Must be within distance threshold AND facing it
        if dist > self.facing_distance_threshold + 1.0:
            return False

        if not self._is_facing_position(pos):
            return False

        # For bin, check that the line of sight is not blocked by an obstacle
        # (box or obstacle ball). This mirrors the real robot's bin_visibility_checker.
        if obj == "bin_1":
            results = self.space.segment_query(
                robot_pos, pos, ROBOT_RADIUS * 0.5,
                pymunk.ShapeFilter(mask=CAT_OBJECT)  # only check movable objects, not walls
            )
            if len(results) > 0:
                return False

        return True

    def _get_object_position(self, obj: str) -> Optional[Tuple[float, float]]:
        """Get the world position of a named object."""
        if obj in ("ball_1", "can_1", "generic_object"):
            if self._ball_picked or self._ball_in_bin:
                return None
            return tuple(self.ball_body.position)
        elif obj == "bin_1":
            return self.config.bin_position
        elif obj == "doorway_1":
            return (self.config.room_width, self.config.doorway_y_center)
        elif obj == "table":
            return (1.5, 0.3)
        return None

    def _is_path_blocked(self, start, end) -> bool:
        """Simple segment query to check if path between two points is blocked."""
        results = self.space.segment_query(
            start, end, ROBOT_RADIUS,
            pymunk.ShapeFilter(mask=CAT_WALL | CAT_OBJECT | CAT_CURTAIN)
        )
        return len(results) > 0

    # ============================================================
    # Debug / Inspection
    # ============================================================

    def get_state_summary(self) -> dict:
        """Return a human-readable state summary for debugging."""
        robot_pos = self.robot_body.position
        heading_deg = math.degrees(self.robot_body.angle)

        state = {
            "robot_position": (round(robot_pos[0], 3), round(robot_pos[1], 3)),
            "robot_heading_deg": round(heading_deg, 1),
            "robot_holding": self._robot_holding,
            "ball_in_bin": self._ball_in_bin,
            "robot_room": "room_1" if self.query_at("robot_1", "room_1") else
                          "room_2" if self.query_at("robot_1", "room_2") else "unknown",
        }

        if not self._ball_picked and not self._ball_in_bin:
            ball_pos = self.ball_body.position
            state["ball_position"] = (round(ball_pos[0], 3), round(ball_pos[1], 3))

        if self.box_body is not None:
            box_pos = self.box_body.position
            state["box_position"] = (round(box_pos[0], 3), round(box_pos[1], 3))

        if self.obstacle_ball_body is not None:
            ob_pos = self.obstacle_ball_body.position
            state["obstacle_ball_position"] = (round(ob_pos[0], 3), round(ob_pos[1], 3))

        if self.curtain_body is not None:
            state["curtain_angle_deg"] = round(math.degrees(self.curtain_body.angle), 1)

        return state
