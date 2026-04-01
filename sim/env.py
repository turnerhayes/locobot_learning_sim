"""
RecycleBotSimEnv: Gymnasium environment wrapping SimWorld.

This provides the same observation/action interface as the real RecycleBotSMDP
but backed by the Pymunk simulation. It can be used standalone for RL training
or integrated with the HybridAgent.

Observation space:
  - Local occupancy grid (8x8 = 64 floats)
  - is_obstructed (1 float)
  - Relative poses to 5 waypoints (10 floats)
  - Symbolic state: room encoding (2), holding (1), facing encoding (5) = 8 floats
  Total: 83 floats (matching real robot)

Action space (primitive only, matching curtain novelty config):
  0: move_forward
  1: turn_left
  2: turn_right

Optionally includes symbolic actions (approach, pick, place, pass_through_door).
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, Set, List

from .world import SimWorld, WorldConfig, NoveltyType


# Waypoint names used for relative pose computation (must match real robot)
TARGET_OBJECTS = ["bin", "generic_object", "table", "atdoor", "postdoor"]

# Facing options for symbolic encoding (must match real robot SymbolicState)
FACING_OPTIONS = ["generic_object", "table", "bin_1", "doorway_1", "nothing"]

# Room options
ROOM_OPTIONS = ["room_1", "room_2"]


class RecycleBotSimEnv(gym.Env):
    """
    Gymnasium environment for the RecycleBot simulation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        novelty: NoveltyType = NoveltyType.NONE,
        failed_operator_name: Optional[str] = None,
        failed_operator_params: Optional[list] = None,
        target_predicates: Optional[Set[tuple]] = None,
        include_local_view: bool = True,
        include_symbolic_actions: bool = False,
        local_view_size: int = 8,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        world_config: Optional[WorldConfig] = None,
    ):
        super().__init__()

        # Build world config
        if world_config is None:
            world_config = WorldConfig(novelty=novelty)
        self.world = SimWorld(world_config)

        self.novelty = novelty
        self.failed_operator_name = failed_operator_name
        self.failed_operator_params = failed_operator_params or []
        self.target_predicates = target_predicates or set()
        self.include_local_view = include_local_view
        self.include_symbolic_actions = include_symbolic_actions
        self.local_view_size = local_view_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Build action list
        self._build_action_list()

        # Define spaces
        obs_size = self._compute_obs_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.action_list))

        self._step_count = 0

    def _build_action_list(self):
        """Build the flat action list matching RecycleBotSMDP's ActionSpace."""
        self.action_list = []

        # Primitive actions (always included)
        self.action_list.append({"type": "primitive", "name": "move_forward"})
        self.action_list.append({"type": "primitive", "name": "turn_left"})
        self.action_list.append({"type": "primitive", "name": "turn_right"})

        if self.include_symbolic_actions:
            # Add grounded symbolic actions (filtered, matching real robot)
            symbolic = self._generate_filtered_symbolic_actions()
            for name, params in symbolic:
                self.action_list.append({
                    "type": "symbolic",
                    "name": name,
                    "params": params,
                })

    def _generate_filtered_symbolic_actions(self) -> List[Tuple[str, list]]:
        """Generate and filter symbolic actions, matching RecycleBotSMDP logic."""
        objects = {
            "object": ["ball_1"],
            "room": ["room_1", "room_2"],
            "container": ["bin_1"],
            "doorway": ["doorway_1"],
        }

        actions = []

        for obj in objects["object"] + objects["container"] + objects["doorway"]:
            for room in objects["room"]:
                actions.append(("approach", [obj, room, "nothing"]))

        for obj in objects["object"]:
            for room in objects["room"]:
                actions.append(("pick", [obj, room]))

        for obj in objects["object"]:
            for room in objects["room"]:
                for container in objects["container"]:
                    actions.append(("place", [obj, room, container]))

        for r1 in objects["room"]:
            for r2 in objects["room"]:
                if r1 != r2:
                    for doorway in objects["doorway"]:
                        actions.append(("pass_through_door", [r1, r2, doorway]))

        # Filter based on failed operator (skip the action that failed)
        filtered = []
        for name, params in actions:
            if (self.failed_operator_name == name and
                    self.failed_operator_params == params):
                continue
            filtered.append((name, params))

        return filtered

    def _compute_obs_size(self) -> int:
        """Compute the observation vector dimension."""
        size = 0
        if self.include_local_view:
            size += self.local_view_size * self.local_view_size  # occupancy grid
        size += 1  # is_obstructed
        size += len(TARGET_OBJECTS) * 2  # relative poses (x, y) per target
        size += len(ROOM_OPTIONS)  # room encoding
        size += 1  # holding encoding
        size += len(FACING_OPTIONS)  # facing encoding
        return size

    # ============================================================
    # Gymnasium Interface
    # ============================================================

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.world.reset()
        self._step_count = 0
        obs = self._get_observation()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute an action and return (obs, reward, terminated, truncated, info).
        """
        self._step_count += 1
        info = {}

        act = self.action_list[action]

        if act["type"] == "primitive":
            self._execute_primitive(act["name"])
        elif act["type"] == "symbolic":
            success = self._execute_symbolic(act["name"], act.get("params", []))
            if not success:
                # Failed precondition
                obs = self._get_observation()
                return obs, -1.0, False, False, {"failure": "precondition_failed"}

        obs = self._get_observation()
        reward, terminated = self._compute_reward()
        truncated = self._step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    def _execute_primitive(self, action_name: str):
        """Execute a primitive action on the sim world."""
        if action_name == "move_forward":
            if not self.world.is_forward_obstructed():
                self.world.move_forward()
        elif action_name == "turn_left":
            self.world.turn_left()
        elif action_name == "turn_right":
            self.world.turn_right()

    def _execute_symbolic(self, action_name: str, params: list) -> bool:
        """Execute a symbolic action. Returns True if preconditions met."""
        if action_name == "approach":
            obj, room, facing = params
            # Check preconditions
            if not self.world.query_at(obj, room):
                return False
            if not self.world.query_at("robot_1", room):
                return False
            # Map to waypoint and navigate
            target = self._map_to_waypoint(obj)
            return self.world.approach_waypoint(target)

        elif action_name == "pick":
            obj, room = params
            if not self.world.query_at(obj, room):
                return False
            if not self.world.query_at("robot_1", room):
                return False
            if not self.world.query_facing(obj):
                return False
            if not self.world.query_hold("nothing"):
                return False
            return self.world.pick_object()

        elif action_name == "place":
            obj, room, container = params
            if not self.world.query_at("robot_1", room):
                return False
            if not self.world.query_at(container, room):
                return False
            if not self.world.query_facing(container):
                return False
            if not self.world.query_hold(obj):
                return False
            return self.world.place_object()

        elif action_name == "pass_through_door":
            room1, room2, doorway = params
            if not self.world.query_at("robot_1", room1):
                return False
            if not self.world.query_facing(doorway):
                return False
            return self.world.approach_waypoint("postdoor")

        return False

    @staticmethod
    def _map_to_waypoint(obj: str) -> str:
        """Map PDDL object name to waypoint name."""
        if obj in ("ball_1", "can_1"):
            return "generic_object"
        if obj == "doorway_1":
            return "atdoor"
        if obj == "bin_1":
            return "bin"
        return obj

    # ============================================================
    # Observation
    # ============================================================

    def _get_observation(self) -> np.ndarray:
        """Build the full observation vector matching the real robot's format."""
        parts = []

        # Sub-symbolic: local occupancy grid
        if self.include_local_view:
            grid = self.world.get_local_occupancy_grid(
                size=self.local_view_size,
                resolution=0.1,
            )
            parts.append(grid.flatten())

        # Sub-symbolic: is_obstructed
        obstructed = 1.0 if self.world.is_forward_obstructed() else 0.0
        parts.append(np.array([obstructed], dtype=np.float32))

        # Sub-symbolic: relative poses
        rel_poses = []
        for target in TARGET_OBJECTS:
            rx, ry = self.world.get_relative_pose(target)
            rel_poses.extend([rx, ry])
        parts.append(np.array(rel_poses, dtype=np.float32))

        # Symbolic: room encoding
        room_enc = np.zeros(len(ROOM_OPTIONS), dtype=np.float32)
        for idx, room in enumerate(ROOM_OPTIONS):
            if self.world.query_at("robot_1", room):
                room_enc[idx] = 1.0
                break
        parts.append(room_enc)

        # Symbolic: holding encoding
        holding = 1.0 if self.world.query_hold("ball_1") else 0.0
        parts.append(np.array([holding], dtype=np.float32))

        # Symbolic: facing encoding
        facing_enc = np.zeros(len(FACING_OPTIONS), dtype=np.float32)
        for idx, obj in enumerate(FACING_OPTIONS):
            if self.world.query_facing(obj):
                facing_enc[idx] = 1.0
                break
        parts.append(facing_enc)

        return np.concatenate(parts, axis=0)

    # ============================================================
    # Reward (mirrors RewardFunction logic)
    # ============================================================

    def _compute_reward(self) -> Tuple[float, bool]:
        """
        Compute reward based on target predicates (plannable states).

        Sparse reward:
          +10.0 when target_predicates ⊆ current_predicates (goal reached)
          -1.0 for failed symbolic action preconditions (handled in step())
           0.0 otherwise
        """
        current_predicates = self._get_current_predicates()

        reward = 0.0
        done = False

        # Success: all target predicates are satisfied in current state
        if (len(self.target_predicates) > 0 and
                len(current_predicates) > 0 and
                self.target_predicates.issubset(current_predicates)):
            reward += 10.0
            done = True

        return reward, done

    def _get_current_predicates(self) -> Set[tuple]:
        """Get the current symbolic predicates as a set of tuples."""
        predicates = set()

        for room in ROOM_OPTIONS:
            if self.world.query_at("robot_1", room):
                predicates.add(("at", room, "robot_1"))
                break

        if self.world.query_hold("ball_1"):
            predicates.add(("hold", "ball_1"))

        for obj in FACING_OPTIONS:
            if self.world.query_facing(obj):
                predicates.add(("facing", obj))
                break

        return predicates

    def _get_distance_to_goal(self) -> Optional[float]:
        """
        Compute distance to the goal implied by the target predicates.
        """
        targets = []
        for pred in self.target_predicates:
            pred_list = list(pred)
            pred_name = pred_list[0]
            if pred_name == "facing" and pred_list[1] != "nothing":
                # Map PDDL name to waypoint name
                targets.append(self._map_to_waypoint(pred_list[1]))
            elif pred_name == "hold" and pred_list[1] != "nothing":
                # Distance to the object we need to pick up
                targets.append(self._map_to_waypoint(pred_list[1]))
            elif pred_name == "at" and pred_list[1].startswith("room_"):
                room = pred_list[1]
                for wp_name in TARGET_OBJECTS:
                    wp_data = self.world._get_waypoints().get(wp_name)
                    if wp_data is None:
                        continue
                    wp_pos = wp_data[0]
                    from shapely.geometry import Point, Polygon
                    boundary = self.world.room_boundaries.get(room)
                    if boundary and Point(wp_pos[0], wp_pos[1]).within(Polygon(boundary)):
                        targets.append(wp_name)

        if not targets:
            return None

        robot_pos = self.world.get_robot_position()
        distances = []
        for target in targets:
            rel = self.world.get_relative_pose(target)
            dist = math.sqrt(rel[0] ** 2 + rel[1] ** 2)
            distances.append(dist)

        return min(distances) if distances else None

    # ============================================================
    # Rendering
    # ============================================================

    def render(self):
        """Render the environment. Delegates to pygame renderer if available."""
        if self.render_mode == "human":
            self._render_pygame()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_pygame(self):
        """Pygame-based real-time rendering."""
        try:
            from .renderer import PygameRenderer
        except ImportError:
            return

        if not hasattr(self, "_renderer"):
            self._renderer = PygameRenderer(self.world)
        self._renderer.draw()

    def _render_rgb_array(self) -> np.ndarray:
        """Return an RGB numpy array of the current state."""
        # Simple matplotlib-based rendering for headless mode
        return self._render_to_array()

    def _render_to_array(self) -> np.ndarray:
        """Render to a numpy array using matplotlib."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle, FancyArrow
        import matplotlib.transforms as transforms
        from .world import ROBOT_RADIUS, BALL_RADIUS, BOX_SIZE, CURTAIN_WIDTH, CURTAIN_LENGTH

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        rw = self.world.config.room_width
        rh = self.world.config.room_height

        # Walls
        ax.plot([0, 2 * rw], [0, 0], "k-", linewidth=2)
        ax.plot([0, 0], [0, rh], "k-", linewidth=2)
        ax.plot([0, 2 * rw], [rh, rh], "k-", linewidth=2)
        ax.plot([2 * rw, 2 * rw], [0, rh], "k-", linewidth=2)

        dy = self.world.config.doorway_y_center
        dhw = self.world.config.doorway_half_width
        ax.plot([rw, rw], [0, dy - dhw], "k-", linewidth=2)
        ax.plot([rw, rw], [dy + dhw, rh], "k-", linewidth=2)

        # Robot
        rp = self.world.get_robot_position()
        heading = self.world.get_robot_heading()
        robot_circle = Circle(rp, ROBOT_RADIUS, fc="steelblue", ec="navy", alpha=0.8)
        ax.add_patch(robot_circle)
        arrow_len = ROBOT_RADIUS * 1.5
        ax.arrow(
            rp[0], rp[1],
            arrow_len * math.cos(heading),
            arrow_len * math.sin(heading),
            head_width=0.08, head_length=0.04, fc="navy", ec="navy",
        )

        # Ball
        if not self.world._ball_picked and not self.world._ball_in_bin:
            bp = tuple(self.world.ball_body.position)
            ball_circle = Circle(bp, BALL_RADIUS, fc="orange", ec="darkorange")
            ax.add_patch(ball_circle)

        # Bin
        bin_pos = self.world.config.bin_position
        bin_rect = Rectangle(
            (bin_pos[0] - 0.15, bin_pos[1] - 0.15), 0.3, 0.3,
            fc="lightgreen" if self.world._ball_in_bin else "lightgray",
            ec="green", linewidth=2,
        )
        ax.add_patch(bin_rect)
        ax.text(bin_pos[0], bin_pos[1], "BIN", ha="center", va="center", fontsize=7)

        # Box (if present)
        if self.world.box_body is not None:
            bp = tuple(self.world.box_body.position)
            box_rect = Rectangle(
                (bp[0] - BOX_SIZE[0] / 2, bp[1] - BOX_SIZE[1] / 2),
                BOX_SIZE[0], BOX_SIZE[1],
                fc="saddlebrown", ec="black", alpha=0.7,
            )
            ax.add_patch(box_rect)

        # Curtain (if present)
        if self.world.curtain_body is not None:
            cp = tuple(self.world.curtain_body.position)
            angle = self.world.curtain_body.angle
            curtain_rect = Rectangle(
                (-CURTAIN_WIDTH / 2, -CURTAIN_LENGTH / 2),
                CURTAIN_WIDTH, CURTAIN_LENGTH,
                fc="purple", ec="indigo", alpha=0.6,
            )
            t = transforms.Affine2D().rotate(angle).translate(cp[0], cp[1]) + ax.transData
            curtain_rect.set_transform(t)
            ax.add_patch(curtain_rect)

        ax.set_xlim(-0.5, 2 * rw + 0.5)
        ax.set_ylim(-0.5, rh + 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"Step {self._step_count}")
        ax.grid(True, alpha=0.3)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        data = np.asarray(buf)[:, :, :3].copy()  # RGBA -> RGB
        plt.close(fig)
        return data

    def close(self):
        if hasattr(self, "_renderer"):
            self._renderer.close()
