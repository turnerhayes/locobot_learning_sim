"""
Comprehensive audit of observation space, action space, reward function, and all calculations.
Traces every value with concrete numbers to catch bugs.
"""

from typing import cast
import sys, os, math
from gymnasium import spaces
import numpy as np
from pymunk import Body
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.world import SimWorld, WorldConfig, NoveltyType, ROBOT_RADIUS, BALL_RADIUS
from sim.env import RecycleBotSimEnv, TARGET_OBJECTS, FACING_OPTIONS, ROOM_OPTIONS

FAIL = False

def check(condition, msg):
    global FAIL
    if not condition:
        print(f"  [BUG] {msg}")
        FAIL = True
    return condition


def audit_observation_space():
    """Verify every dimension of the observation vector is correct."""
    print("\n=== AUDIT: Observation Space ===")

    env = RecycleBotSimEnv(
        novelty=NoveltyType.NONE,
        target_predicates={("at", "room_2", "robot_1")},
        include_local_view=True,
        local_view_size=8,
    )
    obs, _ = env.reset()

    # Expected size: 64 (grid) + 1 (obstruction) + 10 (poses) + 2 (room) + 1 (hold) + 5 (facing) = 83
    expected_size = 64 + 1 + 10 + 2 + 1 + 5
    check(obs.shape == (expected_size,), f"Obs shape should be ({expected_size},), got {obs.shape}")
    check(obs.dtype == np.float32, f"Obs dtype should be float32, got {obs.dtype}")

    # Parse each section
    idx = 0

    # 1. Local occupancy grid: 64 floats
    grid_flat = obs[idx:idx+64]
    idx += 64
    check(grid_flat.min() >= 0.0, f"Grid min should be >= 0, got {grid_flat.min()}")
    check(grid_flat.max() <= 1.0, f"Grid max should be <= 1, got {grid_flat.max()}")
    print(f"  Grid [0:64]: min={grid_flat.min():.2f}, max={grid_flat.max():.2f}, occupied={np.sum(grid_flat > 0)}")

    # 2. is_obstructed: 1 float
    obstructed = obs[idx]
    idx += 1
    check(obstructed in (0.0, 1.0), f"Obstructed should be 0 or 1, got {obstructed}")
    print(f"  Obstructed [{idx-1}]: {obstructed}")

    # 3. Relative poses: 10 floats (5 waypoints × 2)
    for i, target in enumerate(TARGET_OBJECTS):
        rx = obs[idx]
        ry = obs[idx+1]
        idx += 2
        # Verify against world directly
        world_rx, world_ry = env.world.get_relative_pose(target)
        check(abs(rx - world_rx) < 1e-4, f"RelPose {target} x mismatch: obs={rx}, world={world_rx}")
        check(abs(ry - world_ry) < 1e-4, f"RelPose {target} y mismatch: obs={ry}, world={world_ry}")
        print(f"  RelPose[{target}]: ({rx:.3f}, {ry:.3f})")

    # 4. Room encoding: 2 floats
    room_enc = obs[idx:idx+2]
    idx += 2
    check(sum(room_enc) == 1.0, f"Room encoding should sum to 1, got {sum(room_enc)} => {room_enc}")
    in_room1 = env.world.query_at("robot_1", "room_1")
    in_room2 = env.world.query_at("robot_1", "room_2")
    check(room_enc[0] == (1.0 if in_room1 else 0.0), f"Room1 enc mismatch: obs={room_enc[0]}, query={in_room1}")
    check(room_enc[1] == (1.0 if in_room2 else 0.0), f"Room2 enc mismatch: obs={room_enc[1]}, query={in_room2}")
    print(f"  Room encoding: {room_enc} (room_1={in_room1}, room_2={in_room2})")

    # 5. Holding: 1 float
    hold_enc = obs[idx]
    idx += 1
    is_holding = env.world.query_hold("ball_1")
    check(hold_enc == (1.0 if is_holding else 0.0), f"Hold mismatch: obs={hold_enc}, query={is_holding}")
    print(f"  Holding: {hold_enc} (hold_ball={is_holding})")

    # 6. Facing: 5 floats
    facing_enc = obs[idx:idx+5]
    idx += 5
    check(sum(facing_enc) == 1.0, f"Facing encoding should sum to 1, got {sum(facing_enc)} => {facing_enc}")
    for i, obj in enumerate(FACING_OPTIONS):
        is_facing = env.world.query_facing(obj)
        if facing_enc[i] == 1.0:
            check(is_facing, f"Facing enc says facing {obj} but query says no")
            print(f"  Facing: {obj} (enc[{i}]=1.0)")
    print(f"  Facing encoding: {facing_enc}")

    check(idx == expected_size, f"Parsed {idx} dims but expected {expected_size}")
    print(f"  Total dimensions parsed: {idx}/{expected_size}")

    # Also verify with include_local_view=False
    env2 = RecycleBotSimEnv(include_local_view=False, target_predicates={("at", "room_2", "robot_1")})
    obs2, _ = env2.reset()
    expected_no_grid = 1 + 10 + 2 + 1 + 5  # 19
    check(obs2.shape == (expected_no_grid,), f"No-grid obs should be ({expected_no_grid},), got {obs2.shape}")
    print(f"  No-grid observation: shape={obs2.shape}")

    env.close()
    env2.close()


def audit_observation_changes_after_action():
    """Verify that observations actually change after actions."""
    print("\n=== AUDIT: Observation Changes After Actions ===")

    env = RecycleBotSimEnv(target_predicates={("at", "room_2", "robot_1")})
    obs0, _ = env.reset()

    # move_forward should change relative poses
    obs1, _, _, _, _ = env.step(0)  # move_forward
    check(not np.array_equal(obs0, obs1), "Observation should change after move_forward")

    # Specifically, relative pose to atdoor should decrease (robot moves right toward it)
    # Poses start at index 65 (64 grid + 1 obstruction)
    pose_start = 65
    atdoor_idx = TARGET_OBJECTS.index("atdoor") * 2 + pose_start
    old_dx = obs0[atdoor_idx]
    new_dx = obs1[atdoor_idx]
    check(new_dx < old_dx, f"After move_forward, atdoor rel_x should decrease: {old_dx:.3f} -> {new_dx:.3f}")
    print(f"  After move_forward: atdoor rel_x {old_dx:.3f} -> {new_dx:.3f}")

    # turn_left should change facing but not position-based components much
    obs2, _, _, _, _ = env.step(1)  # turn_left
    # Room encoding shouldn't change
    room_start = pose_start + 10
    check(np.array_equal(obs1[room_start:room_start+2], obs2[room_start:room_start+2]),
          "Room encoding shouldn't change on turn")
    print(f"  After turn_left: room encoding unchanged")

    env.close()


def audit_action_space():
    """Verify action space matches expected actions."""
    print("\n=== AUDIT: Action Space ===")

    # Primitive only
    env1 = RecycleBotSimEnv(target_predicates=set())
    action_space = cast(spaces.Discrete, env1.action_space)
    check(action_space.n == 3, f"Primitive-only should have 3 actions, got {action_space.n}")
    check(env1.action_list[0]["name"] == "move_forward", f"Action 0 should be move_forward")
    check(env1.action_list[1]["name"] == "turn_left", f"Action 1 should be turn_left")
    check(env1.action_list[2]["name"] == "turn_right", f"Action 2 should be turn_right")
    print(f"  Primitive actions: {[a['name'] for a in env1.action_list]}")

    # With symbolic
    env2 = RecycleBotSimEnv(
        include_symbolic_actions=True,
        failed_operator_name="pick",
        failed_operator_params=["ball_1", "room_1"],
        target_predicates=set(),
    )
    action_space = cast(spaces.Discrete, env2.action_space)
    check(action_space.n > 3, f"Symbolic env should have >3 actions, got {action_space.n}")
    symbolic_actions = [a for a in env2.action_list if a["type"] == "symbolic"]
    print(f"  With symbolic: {action_space.n} total ({len(symbolic_actions)} symbolic)")
    for a in symbolic_actions:
        print(f"    {a['name']}({a['params']})")

    # Verify failed operator is excluded
    for a in env2.action_list:
        if a["type"] == "symbolic" and a["name"] == "pick":
            is_excluded = a["params"] == ["ball_1", "room_1"]
            check(not is_excluded, f"Failed operator pick(ball_1, room_1) should be excluded but found it")
    print(f"  Verified failed operator excluded from action space")

    env1.close()
    env2.close()


def audit_reward_curtain():
    """Trace reward computation for curtain scenario."""
    print("\n=== AUDIT: Reward — Curtain Scenario ===")

    config = WorldConfig(novelty=NoveltyType.CURTAIN, robot_start=(3.7, 1.5), robot_start_heading=0.0)
    env = RecycleBotSimEnv(
        world_config=config,
        target_predicates={("at", "room_2", "robot_1"), ("facing", "nothing")},
        max_steps=60,
    )
    obs, _ = env.reset()

    # Robot at (3.7, 1.5) in room_1
    current = env._get_current_predicates()
    print(f"  Current predicates: {current}")
    check(("at", "room_1", "robot_1") in current, "Robot should be in room_1")

    # Distance to goal (for reference, not used in sparse reward)
    dist = env._get_distance_to_goal()
    print(f"  Distance to goal: {dist}")
    check(dist is not None, "Distance should not be None for curtain scenario")
    check(cast(float, dist) > 0.0, f"Distance should be > 0, got {dist}")

    # Sparse reward: 0 initially
    reward, done = env._compute_reward()
    check(reward == 0.0, f"Sparse reward should be 0 initially, got {reward}")
    check(not done, "Should not be done at start")
    print(f"  Initial reward: {reward:.4f} (sparse)")

    # Now teleport robot to room_2 and check success
    if env.world.robot_body is None:
        raise ValueError("Robot body not initialized")
    env.world.robot_body.position = (5.0, 1.5)
    env.world.robot_body.angle = math.pi / 2  # facing up, should be facing "nothing"
    current2 = env._get_current_predicates()
    print(f"  After teleport to room_2: predicates = {current2}")

    reward2, done2 = env._compute_reward()
    print(f"  Reward after teleport: {reward2:.4f}, done: {done2}")

    # Should succeed if predicates are subset of target
    target = {("at", "room_2", "robot_1"), ("facing", "nothing")}
    is_subset = current2.issubset(target)
    print(f"  Is {current2} subset of {target}? {is_subset}")
    if is_subset:
        check(done2, "Should be done when predicates match target")
        check(reward2 >= 10.0, f"Success reward should include +10 bonus, got {reward2}")

    env.close()


def audit_reward_box():
    """Trace reward computation for box scenario."""
    print("\n=== AUDIT: Reward — Box Scenario ===")

    config = WorldConfig(novelty=NoveltyType.BOX, robot_start=(4.5, 1.5), robot_start_heading=0.0)
    env = RecycleBotSimEnv(
        world_config=config,
        target_predicates={("facing", "bin_1")},
        max_steps=80,
    )
    obs, _ = env.reset()

    # Distance to bin waypoint
    dist = env._get_distance_to_goal()
    print(f"  Distance to bin waypoint: {dist}")
    check(dist is not None, "Distance should not be None for box scenario")
    check(cast(float, dist) > 0.0, f"Distance should be > 0, got {dist}")

    reward, done = env._compute_reward()
    print(f"  Initial reward: {reward:.4f}, done: {done}")
    check(not done, "Should not be done initially")

    # Verify the waypoint mapping: "bin_1" -> "bin"
    mapped = env._map_to_waypoint("bin_1")
    check(mapped == "bin", f"bin_1 should map to 'bin', got '{mapped}'")

    # Check what happens when we get the robot close to and facing the bin
    if env.world.robot_body is None:
        raise ValueError("Robot body not initialized")
    env.world.robot_body.position = (6.1, 1.5)
    env.world.robot_body.angle = 0.0  # facing right toward bin at (6.5, 1.5)
    current = env._get_current_predicates()
    print(f"  Near bin, facing right: predicates = {current}")

    is_facing_bin = env.world.query_facing("bin_1")
    print(f"  query_facing('bin_1') = {is_facing_bin}")

    reward3, done3 = env._compute_reward()
    print(f"  Near-bin reward: {reward3:.4f}, done: {done3}")

    env.close()


def audit_reward_ball_obstacle():
    """Trace reward for ball_obstacle scenario (ball blocks bin, same target as box)."""
    print("\n=== AUDIT: Reward — Ball Obstacle Scenario ===")

    config = WorldConfig(novelty=NoveltyType.BALL_OBSTACLE, robot_start=(4.5, 1.5), robot_start_heading=0.0)
    env = RecycleBotSimEnv(
        world_config=config,
        target_predicates={("facing", "bin_1")},
        max_steps=80,
    )
    obs, _ = env.reset()

    # Obstacle ball should exist
    check(env.world.obstacle_ball_body is not None, "Obstacle ball should exist")
    ob_pos = tuple(cast(Body, env.world.obstacle_ball_body).position)
    print(f"  Obstacle ball position: ({ob_pos[0]:.3f}, {ob_pos[1]:.3f})")

    # Task ball should also exist (in room 1)
    task_pos = tuple(cast(Body, env.world.ball_body).position)
    print(f"  Task ball position: ({task_pos[0]:.3f}, {task_pos[1]:.3f})")

    # Initial reward: sparse, so 0.0 (not facing bin yet)
    reward, done = env._compute_reward()
    print(f"  Initial reward: {reward:.4f}, done: {done}")
    check(not done, "Should not be done initially")

    # Move robot close to bin and verify success triggers
    if env.world.robot_body is None:
        raise ValueError("Robot body not initialized")
    env.world.robot_body.position = (6.1, 1.5)
    env.world.robot_body.angle = 0.0  # facing toward bin
    current = env._get_current_predicates()
    print(f"  Near bin predicates: {current}")

    reward2, done2 = env._compute_reward()
    print(f"  Near-bin reward: {reward2:.4f}, done: {done2}")

    is_facing = env.world.query_facing("bin_1")
    if is_facing:
        check(done2, "Should be done when facing bin")
        check(reward2 >= 10.0, f"Success reward should be >= 10, got {reward2}")

    env.close()


def audit_facing_consistency():
    """Verify facing encoding is consistent between obs and queries."""
    print("\n=== AUDIT: Facing Consistency ===")

    env = RecycleBotSimEnv(target_predicates=set())
    obs, _ = env.reset()

    # Extract facing from observation
    facing_start = 64 + 1 + 10 + 2 + 1  # grid + obstructed + poses + room + hold
    facing_obs = obs[facing_start:facing_start+5]

    # Query each
    for i, obj in enumerate(FACING_OPTIONS):
        query_result = env.world.query_facing(obj)
        obs_result = facing_obs[i] == 1.0
        if query_result or obs_result:
            check(query_result == obs_result,
                  f"Facing '{obj}': obs={obs_result}, query={query_result}")
            print(f"  Facing '{obj}': obs={facing_obs[i]}, query={query_result}")

    # Verify exactly one facing is set
    check(sum(facing_obs) == 1.0, f"Exactly one facing should be 1.0, got sum={sum(facing_obs)}")

    # Turn robot to various angles and verify facing updates
    for angle, expected_facing in [
        (0.0, None),  # facing right, may face ball or doorway depending on distance
        (math.pi, None),  # facing left
        (math.pi/2, None),  # facing up
    ]:
        if env.world.robot_body is None:
            raise ValueError("Robot body not initialized")
        env.world.robot_body.angle = angle
        obs_new = env._get_observation()
        facing_new = obs_new[facing_start:facing_start+5]
        check(sum(facing_new) == 1.0, f"At angle {math.degrees(angle):.0f}°: facing sum should be 1, got {sum(facing_new)}")
        active = [FACING_OPTIONS[i] for i in range(5) if facing_new[i] == 1.0]
        print(f"  Heading {math.degrees(angle):.0f}°: facing={active}")

    env.close()


def audit_room_boundary_edge_cases():
    """Test room boundary queries at edge positions."""
    print("\n=== AUDIT: Room Boundary Edge Cases ===")

    world = SimWorld()

    # Exactly at doorway x=4.0 — could be in either room
    if world.robot_body is None:
        raise ValueError("Robot body not initialized")
    world.robot_body.position = (4.0, 1.5)
    r1 = world.query_at("robot_1", "room_1")
    r2 = world.query_at("robot_1", "room_2")
    print(f"  At doorway (4.0, 1.5): room_1={r1}, room_2={r2}")
    # Shapely point_within on boundary returns False, so robot might be in neither
    # This is fine as long as it's consistent

    # Just inside room_1
    world.robot_body.position = (3.99, 1.5)
    check(world.query_at("robot_1", "room_1"), "At x=3.99 should be in room_1")
    print(f"  At (3.99, 1.5): room_1={world.query_at('robot_1', 'room_1')}")

    # Just inside room_2
    world.robot_body.position = (4.01, 1.5)
    check(world.query_at("robot_1", "room_2"), "At x=4.01 should be in room_2")
    print(f"  At (4.01, 1.5): room_2={world.query_at('robot_1', 'room_2')}")

    # Corner of room
    world.robot_body.position = (0.01, 0.01)
    check(world.query_at("robot_1", "room_1"), "At corner (0.01, 0.01) should be in room_1")
    print(f"  At corner (0.01, 0.01): room_1={world.query_at('robot_1', 'room_1')}")


def audit_step_reward_trajectory():
    """Run a short trajectory and verify rewards are reasonable."""
    print("\n=== AUDIT: Step-by-step Reward Trajectory ===")

    config = WorldConfig(novelty=NoveltyType.CURTAIN, robot_start=(3.7, 1.5), robot_start_heading=0.0)
    env = RecycleBotSimEnv(
        world_config=config,
        target_predicates={("at", "room_2", "robot_1"), ("facing", "nothing")},
        max_steps=60,
    )
    obs, _ = env.reset()

    prev_reward = None
    for step in range(10):
        obs, reward, terminated, truncated, info = env.step(0)  # move_forward
        pos = env.world.get_robot_position()
        print(f"  Step {step}: pos=({pos[0]:.3f}, {pos[1]:.3f}), reward={reward:.4f}, term={terminated}, trunc={truncated}")
        check(not math.isnan(reward), f"Reward is NaN at step {step}")
        check(not math.isinf(reward), f"Reward is Inf at step {step}")
        check(not truncated, f"Should not be truncated at step {step}")
        prev_reward = reward

    env.close()


def audit_reset_consistency():
    """Verify reset produces consistent initial state."""
    print("\n=== AUDIT: Reset Consistency ===")

    env = RecycleBotSimEnv(target_predicates={("at", "room_2", "robot_1")})

    obs1, _ = env.reset()
    obs2, _ = env.reset()
    check(np.array_equal(obs1, obs2), "Two resets should produce identical observations")
    print(f"  Two resets match: {np.array_equal(obs1, obs2)}")

    # Step, then reset — should go back to initial
    env.step(0)
    env.step(1)
    obs3, _ = env.reset()
    check(np.array_equal(obs1, obs3), "Reset after steps should restore initial state")
    print(f"  Reset after steps matches initial: {np.array_equal(obs1, obs3)}")

    env.close()


def audit_terminated_vs_truncated():
    """Verify terminated and truncated flags are correct."""
    print("\n=== AUDIT: Terminated vs Truncated ===")

    env = RecycleBotSimEnv(
        target_predicates={("at", "room_2", "robot_1")},
        max_steps=5,
    )
    obs, _ = env.reset()

    for step in range(10):
        obs, reward, terminated, truncated, info = env.step(1)  # just turn, never reach goal
        if truncated:
            check(step == 4, f"Should truncate at step 4 (max_steps=5), got step {step}")
            check(not terminated, "Truncated should not also be terminated")
            print(f"  Correctly truncated at step {step}")
            break

    env.close()


def main():
    global FAIL

    audits = [
        audit_observation_space,
        audit_observation_changes_after_action,
        audit_action_space,
        audit_reward_curtain,
        audit_reward_box,
        audit_reward_ball_obstacle,
        audit_facing_consistency,
        audit_room_boundary_edge_cases,
        audit_step_reward_trajectory,
        audit_reset_consistency,
        audit_terminated_vs_truncated,
    ]

    for audit in audits:
        try:
            audit()
        except Exception as e:
            print(f"  [CRASH] {audit.__name__}: {e}")
            import traceback
            traceback.print_exc()
            FAIL = True

    print("\n" + "=" * 60)
    if FAIL:
        print("AUDIT RESULT: BUGS FOUND — see [BUG] lines above")
    else:
        print("AUDIT RESULT: ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
