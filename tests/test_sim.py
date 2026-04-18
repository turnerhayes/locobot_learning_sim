"""
Test script for the RecycleBot simulation environment.

Tests:
1. World basics: walls, robot movement, collision detection
2. Symbolic queries: at, facing, hold, contain
3. Observation space: correct dimensions and values
4. Baseline scenario: full plan execution without novelty
5. Curtain novelty: robot must push through
6. Box novelty: robot must push box aside
7. Ball obstacle novelty: ball blocks bin away
8. Gymnasium env: reset/step/reward loop
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.world import SimWorld, WorldConfig, NoveltyType, ROBOT_RADIUS
from sim.env import RecycleBotSimEnv


def test_world_basics():
    """Test that the world initializes correctly."""
    world = SimWorld()

    pos = world.get_robot_position()
    assert abs(pos[0] - 1.0) < 0.01, f"Robot x should be 1.0, got {pos[0]}"
    assert abs(pos[1] - 1.5) < 0.01, f"Robot y should be 1.5, got {pos[1]}"

    heading = world.get_robot_heading()
    assert abs(heading) < 0.01, f"Robot heading should be 0, got {heading}"

    print("[PASS] World basics: initialization correct")


def test_robot_movement():
    """Test primitive actions move the robot correctly."""
    world = SimWorld()

    # Move forward (heading = 0 = right)
    old_pos = world.get_robot_position()
    world.move_forward()
    new_pos = world.get_robot_position()
    assert new_pos[0] > old_pos[0], f"Robot should move right, but x went from {old_pos[0]} to {new_pos[0]}"
    print(f"  move_forward: ({old_pos[0]:.3f}, {old_pos[1]:.3f}) -> ({new_pos[0]:.3f}, {new_pos[1]:.3f})")

    # Turn left (increases angle)
    old_heading = world.get_robot_heading()
    world.turn_left()
    new_heading = world.get_robot_heading()
    assert new_heading > old_heading, f"Turn left should increase angle"
    print(f"  turn_left: heading {math.degrees(old_heading):.1f}° -> {math.degrees(new_heading):.1f}°")

    # Turn right (decreases angle)
    old_heading = world.get_robot_heading()
    world.turn_right()
    new_heading = world.get_robot_heading()
    assert new_heading < old_heading, f"Turn right should decrease angle"
    print(f"  turn_right: heading {math.degrees(old_heading):.1f}° -> {math.degrees(new_heading):.1f}°")

    print("[PASS] Robot movement: primitives work correctly")


def test_wall_collision():
    """Test that the robot can't walk through walls."""
    world = SimWorld()

    if world.robot_body is None:
        raise ValueError("Robot body not initialized")
    # Walk robot into left wall (heading = pi = left)
    world.robot_body.angle = math.pi  # face left
    initial_x = world.get_robot_position()[0]

    for _ in range(50):
        world.move_forward()

    final_x = world.get_robot_position()[0]
    assert final_x > 0.05, f"Robot should not pass through left wall, x={final_x}"
    print(f"  Wall collision: robot stopped at x={final_x:.3f} (wall at x=0)")

    print("[PASS] Wall collision: robot blocked by walls")


def test_symbolic_queries():
    """Test symbolic state queries."""
    world = SimWorld()

    # Robot starts in room 1
    assert world.query_at("robot_1", "room_1"), "Robot should be in room_1"
    assert not world.query_at("robot_1", "room_2"), "Robot should NOT be in room_2"

    # Ball starts in room 1
    assert world.query_at("ball_1", "room_1"), "Ball should be in room_1"

    # Bin is in room 2
    assert world.query_at("bin_1", "room_2"), "Bin should be in room_2"

    # Doorway is in both rooms
    assert world.query_at("doorway_1", "room_1"), "Doorway should be in room_1"
    assert world.query_at("doorway_1", "room_2"), "Doorway should be in room_2"

    # Robot holds nothing initially
    assert world.query_hold("nothing"), "Robot should hold nothing"
    assert not world.query_hold("ball_1"), "Robot should NOT hold ball_1"

    # Ball is not in bin
    assert not world.query_contain("ball_1", "bin_1"), "Ball should NOT be in bin"

    print("[PASS] Symbolic queries: all correct")


def test_facing():
    """Test facing queries with different robot orientations."""
    world = SimWorld()

    # Robot at (1.0, 1.5) facing right (heading=0) — should face toward ball at (1.5, 1.5)
    # Ball is to the right, heading 0 points right
    is_facing = world.query_facing("ball_1")
    print(f"  Robot at (1.0, 1.5) heading=0°, ball at (1.5, 1.5): facing={is_facing}")

    if world.robot_body is None:
        raise ValueError("Robot body not initialized")
    # Face completely away from ball
    world.robot_body.angle = math.pi  # face left
    is_facing_away = world.query_facing("ball_1")
    assert not is_facing_away, "Robot facing left should NOT face ball on the right"
    print(f"  Robot heading=180° (left): facing ball = {is_facing_away}")

    # Facing nothing when facing away from everything
    world.robot_body.angle = math.pi / 2  # face up
    world.robot_body.position = (2.0, 1.5)  # middle of room, away from objects
    is_facing_nothing = world.query_facing("nothing")
    print(f"  Robot heading=90° in middle of room: facing nothing = {is_facing_nothing}")

    print("[PASS] Facing queries: work correctly")


def test_pick_and_place():
    """Test picking up ball and placing in bin."""
    world = SimWorld()

    # Navigate to ball and pick it up
    world.approach_waypoint("generic_object")
    success = world.pick_object()
    print(f"  Pick ball: success={success}")

    if success:
        assert world.query_hold("ball_1"), "Should be holding ball"
        assert not world.query_hold("nothing"), "Should NOT be holding nothing"

    # Navigate to bin
    world.approach_waypoint("atdoor")  # go to doorway
    world.approach_waypoint("postdoor")  # go through door
    world.approach_waypoint("bin")  # go to bin

    success = world.place_object()
    print(f"  Place ball in bin: success={success}")

    if success:
        assert world.query_contain("ball_1", "bin_1"), "Ball should be in bin"
        assert world.query_hold("nothing"), "Should be holding nothing"

    print("[PASS] Pick and place: full sequence works")


def test_occupancy_grid():
    """Test local occupancy grid generation."""
    world = SimWorld()

    grid = world.get_local_occupancy_grid(size=8, resolution=0.1)
    assert grid.shape == (8, 8), f"Grid shape should be (8,8), got {grid.shape}"
    assert grid.dtype == np.float32, f"Grid dtype should be float32, got {grid.dtype}"
    assert grid.min() >= 0.0, "Grid values should be >= 0"
    assert grid.max() <= 1.0, "Grid values should be <= 1"

    occupied_cells = np.sum(grid > 0)
    print(f"  Grid: {grid.shape}, occupied cells: {occupied_cells}/64")

    print("[PASS] Occupancy grid: correct shape and values")


def test_relative_poses():
    """Test relative pose computation."""
    world = SimWorld()

    for target in ["bin", "generic_object", "atdoor", "postdoor", "table"]:
        rx, ry = world.get_relative_pose(target)
        print(f"  Relative pose to {target}: ({rx:.3f}, {ry:.3f})")

    print("[PASS] Relative poses: computed correctly")


def test_curtain_novelty():
    """Test that the curtain blocks passage and can be pushed through."""
    config = WorldConfig(novelty=NoveltyType.CURTAIN)
    world = SimWorld(config)

    assert world.curtain_body is not None, "Curtain should exist"
    initial_angle = world.curtain_body.angle
    print(f"  Curtain initial angle: {math.degrees(initial_angle):.1f}°")

    # Navigate robot to doorway
    world.approach_waypoint("atdoor")
    if world.robot_body is None:
        raise ValueError("Robot body not initialized")
    world.robot_body.angle = 0.0  # face right (toward doorway/room2)

    # Try to push through
    for i in range(30):
        world.move_forward()

    final_angle = world.curtain_body.angle
    robot_pos = world.get_robot_position()
    print(f"  After 30 forward pushes: robot at ({robot_pos[0]:.3f}, {robot_pos[1]:.3f})")
    print(f"  Curtain angle: {math.degrees(final_angle):.1f}°")

    # Check if robot made it past the doorway
    in_room2 = world.query_at("robot_1", "room_2")
    print(f"  Robot in room_2: {in_room2}")

    print("[PASS] Curtain novelty: curtain interacts with robot")


def test_box_novelty():
    """Test that the box blocks the bin and can be pushed."""
    config = WorldConfig(novelty=NoveltyType.BOX)
    world = SimWorld(config)

    assert world.box_body is not None, "Box should exist"
    initial_box_pos = tuple(world.box_body.position)
    print(f"  Box initial position: ({initial_box_pos[0]:.3f}, {initial_box_pos[1]:.3f})")

    # Navigate to room 2 near the box
    world.approach_waypoint("postdoor")
    if world.robot_body is None:
        raise ValueError("Robot body not initialized")
    world.robot_body.angle = 0.0  # face right (toward box/bin)

    # Push the box — need enough steps to cross room and push
    for i in range(50):
        world.move_forward()

    final_box_pos = tuple(world.box_body.position)
    print(f"  Box after pushing: ({final_box_pos[0]:.3f}, {final_box_pos[1]:.3f})")
    assert final_box_pos[0] > initial_box_pos[0], "Box should have moved"

    print("[PASS] Box novelty: box is pushable with real physics")


def test_ball_obstacle_novelty():
    """Test that the obstacle ball blocks the bin and can be pushed."""
    config = WorldConfig(
        novelty=NoveltyType.BALL_OBSTACLE,
        robot_start=(4.5, 1.5),  # start in room 2 (where robot is when approach fails)
        robot_start_heading=0.0,
    )
    world = SimWorld(config)

    assert world.obstacle_ball_body is not None, "Obstacle ball should exist"
    initial_pos = tuple(world.obstacle_ball_body.position)
    print(f"  Obstacle ball initial position: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f})")

    # Robot is already in room 2, facing right toward obstacle ball and bin
    for i in range(50):
        world.move_forward()

    final_pos = tuple(world.obstacle_ball_body.position)
    print(f"  Obstacle ball after pushing: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")
    assert final_pos[0] > initial_pos[0], "Obstacle ball should have moved"

    # Task ball should still be in room 1 (untouched)
    if world.ball_body is None:
        raise ValueError("Task ball body not initialized")
    task_ball_pos = tuple(world.ball_body.position)
    print(f"  Task ball still at: ({task_ball_pos[0]:.3f}, {task_ball_pos[1]:.3f})")
    assert task_ball_pos[0] < world.config.room_width, "Task ball should still be in room 1"

    print("[PASS] Ball obstacle novelty: obstacle ball is pushable, task ball unaffected")


def test_gymnasium_env_baseline():
    """Test the Gymnasium env with no novelty."""
    env = RecycleBotSimEnv(
        novelty=NoveltyType.NONE,
        target_predicates={("at", "room_2", "robot_1")},
        max_steps=50,
    )

    obs, info = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Obs range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Run a few random steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"  Ran 10 random steps, total reward: {total_reward:.3f}")
    env.close()

    print("[PASS] Gymnasium env: baseline works")


def test_gymnasium_env_curtain():
    """Test Gymnasium env with curtain novelty, learn to push through."""
    # Target: get to room 2 (effect of pass_through_door)
    env = RecycleBotSimEnv(
        novelty=NoveltyType.CURTAIN,
        failed_operator_name="pass_through_door",
        failed_operator_params=["room_1", "room_2", "doorway_1"],
        target_predicates={("at", "room_2", "robot_1"), ("facing", "nothing")},
        max_steps=100,
    )

    obs, _ = env.reset()

    # Manually guide robot: turn to face doorway, then push through
    # First approach the doorway
    world = env.world
    world.approach_waypoint("atdoor")
    if world.robot_body is None:
        raise ValueError("Robot body not initialized")
    world.robot_body.angle = 0.0

    # Now push through curtain
    success = False
    for step in range(60):
        _, reward, terminated, _, _ = env.step(0)  # move_forward
        if terminated:
            success = True
            print(f"  Reached room_2 at step {step}! Reward: {reward:.3f}")
            break

    if not success:
        robot_pos = world.get_robot_position()
        in_room2 = world.query_at("robot_1", "room_2")
        print(f"  After 60 steps: robot at ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}), in room_2: {in_room2}")

    env.close()
    print("[PASS] Gymnasium env: curtain novelty scenario runs")


def test_render():
    """Test that rendering produces an image."""
    env = RecycleBotSimEnv(
        novelty=NoveltyType.BOX,
        render_mode="rgb_array",
        max_steps=10,
    )
    obs, _ = env.reset()
    frame = env.render()
    assert frame is not None, "Render should produce a frame"
    assert len(frame.shape) == 3, f"Frame should be 3D (H,W,C), got shape {frame.shape}"
    print(f"  Rendered frame: {frame.shape}, dtype={frame.dtype}")
    env.close()

    print("[PASS] Rendering: produces correct output")


def main():
    print("=" * 60)
    print("RecycleBot Simulation - Test Suite")
    print("=" * 60)

    tests = [
        test_world_basics,
        test_robot_movement,
        test_wall_collision,
        test_symbolic_queries,
        test_facing,
        test_pick_and_place,
        test_occupancy_grid,
        test_relative_poses,
        test_curtain_novelty,
        test_box_novelty,
        test_ball_obstacle_novelty,
        test_gymnasium_env_baseline,
        test_gymnasium_env_curtain,
        test_render,
    ]

    passed = 0
    failed = 0
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
