"""
Pygame-based real-time renderer for the RecycleBot simulation.

Renders a top-down 2D view of the environment including:
  - Room walls and doorway
  - Robot with heading indicator
  - Ball, bin, box, curtain
  - Local occupancy grid overlay (optional)
  - Step counter and symbolic state info
"""

import math
import sys
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from .world import (
    SimWorld, ROBOT_RADIUS, BALL_RADIUS, BOX_SIZE,
    CURTAIN_WIDTH, CURTAIN_LENGTH, NoveltyType,
)

# Display settings
PIXELS_PER_METER = 80
MARGIN = 40  # pixels of padding around the world
FPS = 30

# Colors
COLOR_BG = (245, 245, 240)
COLOR_WALL = (50, 50, 50)
COLOR_ROOM1_BG = (255, 255, 255)
COLOR_ROOM2_BG = (248, 248, 255)
COLOR_ROBOT = (70, 130, 180)
COLOR_ROBOT_HEADING = (25, 25, 112)
COLOR_BALL = (255, 165, 0)
COLOR_BALL_OUTLINE = (200, 120, 0)
COLOR_BIN_EMPTY = (200, 200, 200)
COLOR_BIN_FULL = (144, 238, 144)
COLOR_BIN_OUTLINE = (34, 139, 34)
COLOR_BOX = (139, 90, 43)
COLOR_BOX_OUTLINE = (80, 50, 20)
COLOR_CURTAIN = (128, 0, 128)
COLOR_DOORWAY = (220, 220, 210)
COLOR_GRID_OCCUPIED = (255, 0, 0, 80)
COLOR_GRID_FREE = (0, 255, 0, 30)
COLOR_TEXT = (30, 30, 30)
COLOR_INFO_BG = (255, 255, 255, 200)


class PygameRenderer:
    """Real-time Pygame renderer for the SimWorld."""

    def __init__(self, world: SimWorld, show_grid: bool = False):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for PygameRenderer. Install with: pip install pygame")

        self.world = world
        self.show_grid = show_grid

        pygame.init()
        pygame.display.set_caption("RecycleBot Simulation")

        rw = world.config.room_width
        rh = world.config.room_height

        self.world_width = 2 * rw
        self.world_height = rh

        self.screen_width = int(self.world_width * PIXELS_PER_METER) + 2 * MARGIN
        self.screen_height = int(self.world_height * PIXELS_PER_METER) + 2 * MARGIN + 60  # extra for info bar

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_large = pygame.font.SysFont("monospace", 16, bold=True)
        self.step_count = 0

    def world_to_screen(self, x: float, y: float) -> tuple:
        """Convert world coordinates to screen pixels. Y is flipped for screen."""
        sx = int(x * PIXELS_PER_METER) + MARGIN
        sy = int((self.world_height - y) * PIXELS_PER_METER) + MARGIN
        return (sx, sy)

    def meters_to_pixels(self, m: float) -> int:
        return max(1, int(m * PIXELS_PER_METER))

    def draw(self, step: int = 0, episode: int = 0, reward: float = 0.0, info: str = ""):
        """Draw one frame of the environment."""
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    sys.exit()

        self.step_count = step
        self.screen.fill(COLOR_BG)

        self._draw_rooms()
        self._draw_doorway()
        self._draw_walls()

        if self.show_grid:
            self._draw_occupancy_grid()

        self._draw_bin()
        self._draw_curtain()
        self._draw_box()
        self._draw_obstacle_ball()
        self._draw_ball()
        self._draw_robot()
        self._draw_info_bar(step, episode, reward, info)

        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_rooms(self):
        """Draw room backgrounds."""
        rw = self.world.config.room_width
        rh = self.world.config.room_height

        # Room 1
        r1_tl = self.world_to_screen(0, rh)
        r1_size = (self.meters_to_pixels(rw), self.meters_to_pixels(rh))
        pygame.draw.rect(self.screen, COLOR_ROOM1_BG, (*r1_tl, *r1_size))

        # Room 2
        r2_tl = self.world_to_screen(rw, rh)
        r2_size = (self.meters_to_pixels(rw), self.meters_to_pixels(rh))
        pygame.draw.rect(self.screen, COLOR_ROOM2_BG, (*r2_tl, *r2_size))

        # Labels
        label1 = self.font.render("Room 1", True, (150, 150, 150))
        label2 = self.font.render("Room 2", True, (150, 150, 150))
        self.screen.blit(label1, self.world_to_screen(0.1, rh - 0.1))
        self.screen.blit(label2, self.world_to_screen(rw + 0.1, rh - 0.1))

    def _draw_doorway(self):
        """Draw the doorway opening."""
        rw = self.world.config.room_width
        dy = self.world.config.doorway_y_center
        dhw = self.world.config.doorway_half_width

        top = self.world_to_screen(rw - 0.05, dy + dhw)
        width = self.meters_to_pixels(0.1)
        height = self.meters_to_pixels(2 * dhw)
        pygame.draw.rect(self.screen, COLOR_DOORWAY, (*top, width, height))

    def _draw_walls(self):
        """Draw wall segments."""
        rw = self.world.config.room_width
        rh = self.world.config.room_height
        dy = self.world.config.doorway_y_center
        dhw = self.world.config.doorway_half_width

        segments = [
            ((0, 0), (rw, 0)),
            ((0, 0), (0, rh)),
            ((0, rh), (rw, rh)),
            ((rw, 0), (2 * rw, 0)),
            ((2 * rw, 0), (2 * rw, rh)),
            ((rw, rh), (2 * rw, rh)),
            ((rw, 0), (rw, dy - dhw)),
            ((rw, dy + dhw), (rw, rh)),
        ]

        for (x1, y1), (x2, y2) in segments:
            p1 = self.world_to_screen(x1, y1)
            p2 = self.world_to_screen(x2, y2)
            pygame.draw.line(self.screen, COLOR_WALL, p1, p2, 3)

    def _draw_robot(self):
        """Draw the robot with heading indicator."""
        pos = self.world.get_robot_position()
        heading = self.world.get_robot_heading()
        center = self.world_to_screen(pos[0], pos[1])
        radius = self.meters_to_pixels(ROBOT_RADIUS)

        # Robot body
        pygame.draw.circle(self.screen, COLOR_ROBOT, center, radius)
        pygame.draw.circle(self.screen, COLOR_ROBOT_HEADING, center, radius, 2)

        # Heading arrow
        arrow_len = radius * 1.6
        # Note: screen Y is inverted, so negate the sin component
        end_x = center[0] + arrow_len * math.cos(heading)
        end_y = center[1] - arrow_len * math.sin(heading)
        pygame.draw.line(self.screen, COLOR_ROBOT_HEADING, center, (int(end_x), int(end_y)), 3)

        # Small dot at the tip
        pygame.draw.circle(self.screen, COLOR_ROBOT_HEADING, (int(end_x), int(end_y)), 4)

        # "Held" indicator
        if self.world._robot_holding is not None:
            held_text = self.font.render(f"[{self.world._robot_holding}]", True, COLOR_BALL)
            self.screen.blit(held_text, (center[0] - 20, center[1] - radius - 18))

    def _draw_ball(self):
        """Draw the ball if it's in the world."""
        if self.world.ball_body is None:
            raise ValueError("Ball body not initialized")
        if self.world._ball_picked or self.world._ball_in_bin:
            return

        pos = tuple(self.world.ball_body.position)
        center = self.world_to_screen(pos[0], pos[1])
        radius = max(4, self.meters_to_pixels(BALL_RADIUS))

        pygame.draw.circle(self.screen, COLOR_BALL, center, radius)
        pygame.draw.circle(self.screen, COLOR_BALL_OUTLINE, center, radius, 2)

    def _draw_bin(self):
        """Draw the bin."""
        bin_pos = self.world.config.bin_position
        center = self.world_to_screen(bin_pos[0], bin_pos[1])
        half_size = self.meters_to_pixels(0.15)

        fill = COLOR_BIN_FULL if self.world._ball_in_bin else COLOR_BIN_EMPTY
        rect = pygame.Rect(center[0] - half_size, center[1] - half_size, 2 * half_size, 2 * half_size)
        pygame.draw.rect(self.screen, fill, rect)
        pygame.draw.rect(self.screen, COLOR_BIN_OUTLINE, rect, 2)

        label = self.font.render("BIN", True, COLOR_BIN_OUTLINE)
        self.screen.blit(label, (center[0] - 12, center[1] - 7))

    def _draw_box(self):
        """Draw the box obstacle if present."""
        if self.world.box_body is None:
            return

        pos = tuple(self.world.box_body.position)
        center = self.world_to_screen(pos[0], pos[1])
        hw = self.meters_to_pixels(BOX_SIZE[0] / 2)
        hh = self.meters_to_pixels(BOX_SIZE[1] / 2)

        rect = pygame.Rect(center[0] - hw, center[1] - hh, 2 * hw, 2 * hh)
        pygame.draw.rect(self.screen, COLOR_BOX, rect)
        pygame.draw.rect(self.screen, COLOR_BOX_OUTLINE, rect, 2)

        label = self.font.render("BOX", True, (255, 255, 255))
        self.screen.blit(label, (center[0] - 12, center[1] - 7))

    def _draw_obstacle_ball(self):
        """Draw the obstacle ball if present."""
        if self.world.obstacle_ball_body is None:
            return

        pos = tuple(self.world.obstacle_ball_body.position)
        center = self.world_to_screen(pos[0], pos[1])
        radius = max(4, self.meters_to_pixels(self.world.config.obstacle_ball_radius))

        pygame.draw.circle(self.screen, (200, 50, 50), center, radius)
        pygame.draw.circle(self.screen, (120, 20, 20), center, radius, 2)

    def _draw_curtain(self):
        """Draw the curtain if present."""
        if self.world.curtain_body is None:
            return

        pos = tuple(self.world.curtain_body.position)
        angle = self.world.curtain_body.angle
        anchor = self.world.config.curtain_anchor

        # Draw as a rotated line from anchor downward
        length = CURTAIN_LENGTH
        # End point in local frame (hangs down = negative y in world, but rotated by angle)
        end_x = anchor[0] + length * math.sin(angle)
        end_y = anchor[1] - length * math.cos(angle)

        p1 = self.world_to_screen(anchor[0], anchor[1])
        p2 = self.world_to_screen(end_x, end_y)

        pygame.draw.line(self.screen, COLOR_CURTAIN, p1, p2, 4)
        pygame.draw.circle(self.screen, COLOR_CURTAIN, p1, 4)  # pivot point

    def _draw_occupancy_grid(self):
        """Overlay the local occupancy grid on the robot's position."""
        grid = self.world.get_local_occupancy_grid(size=8, resolution=0.1)
        robot_pos = self.world.get_robot_position()
        heading = self.world.get_robot_heading()

        cos_a = math.cos(heading)
        sin_a = math.sin(heading)
        half = 4.0
        res = 0.1

        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)

        for gy in range(8):
            for gx in range(8):
                local_x = (gx - half + 0.5) * res
                local_y = (gy - half + 0.5) * res
                world_x = robot_pos[0] + local_x * cos_a - local_y * sin_a
                world_y = robot_pos[1] + local_x * sin_a + local_y * cos_a

                sx, sy = self.world_to_screen(world_x, world_y)
                cell_size = max(2, self.meters_to_pixels(res))

                if grid[gy, gx] > 0.5:
                    color = (255, 0, 0, 80)
                else:
                    color = (0, 255, 0, 30)

                pygame.draw.rect(overlay, color, (sx - cell_size // 2, sy - cell_size // 2, cell_size, cell_size))

        self.screen.blit(overlay, (0, 0))

    def _draw_info_bar(self, step: int, episode: int, reward: float, info: str):
        """Draw info text at the bottom of the screen."""
        y_start = self.screen_height - 55

        # Background bar
        pygame.draw.rect(self.screen, (240, 240, 240), (0, y_start, self.screen_width, 55))
        pygame.draw.line(self.screen, (180, 180, 180), (0, y_start), (self.screen_width, y_start), 1)

        # State info
        pos = self.world.get_robot_position()
        heading_deg = math.degrees(self.world.get_robot_heading())
        room = "room_1" if self.world.query_at("robot_1", "room_1") else \
               "room_2" if self.world.query_at("robot_1", "room_2") else "?"
        holding = self.world._robot_holding or "nothing"

        novelty_str = self.world.config.novelty.value
        line1 = f"Ep:{episode}  Step:{step}  Reward:{reward:+.2f}  Novelty:{novelty_str}"
        line2 = f"Pos:({pos[0]:.2f},{pos[1]:.2f})  Hdg:{heading_deg:.0f}°  Room:{room}  Hold:{holding}"
        if info:
            line2 += f"  {info}"

        text1 = self.font_large.render(line1, True, COLOR_TEXT)
        text2 = self.font.render(line2, True, COLOR_TEXT)
        self.screen.blit(text1, (10, y_start + 5))
        self.screen.blit(text2, (10, y_start + 25))

    def close(self):
        """Clean up pygame."""
        pygame.quit()
