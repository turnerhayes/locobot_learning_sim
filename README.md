# RecycleBot Simulation Environment

A 2D physics-based simulation for testing neurosymbolic novelty handling on a LocoBot-inspired mobile robot. Built with [Pymunk](https://www.pymunk.org/) for real rigid-body physics and [Gymnasium](https://gymnasium.farama.org/) for RL integration.

## Overview

The robot must pick up an object in one room, navigate through a doorway to the other room, and place it in a bin. A symbolic planner (PDDL) generates the plan, but **novelties** — unknown obstacles — cause operators to fail at execution time. The agent must then learn a primitive recovery policy via reinforcement learning.

Three novelties are supported:

1. **Curtain** — blocks the doorway, `pass_through_door` fails. Robot learns to push through.
2. **Box** — blocks the bin, `approach bin_1` fails. Robot learns to push the box aside.
3. **Ball obstacle** — blocks the bin, `approach bin_1` fails. Robot learns to push the ball aside (different physics from box — it rolls).

```
┌─────────────────────┐         ┌─────────────────────┐
│       Room 1        │ doorway │       Room 2        │
│                     │         │                     │
│  [Robot] -> [Ball]  │===||====│  [Box/Ball] [Bin]   │
│                     │ curtain?│                     │
└─────────────────────┘         └─────────────────────┘
```

The simulator provides the **exact same 83-dimensional observation/action interface** as the real robot's `RecycleBotSMDP`, so the same `HybridAgent` architecture can drive both.

---

## Installation

### macOS

```bash
# Install Python 3.9+ if not already present
brew install python@3.11

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

**Note on pygame**: If `pip install pygame` fails on macOS:
```bash
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
pip install pygame
```

**Note on torch**: If you don't need GPU, the CPU-only version is much smaller:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt  # install the rest
```

### Linux (Ubuntu/Debian)

```bash
# System dependencies for pygame
sudo apt-get update
sudo apt-get install -y python3-dev python3-venv libsdl2-dev libsdl2-image-dev \
    libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt
```

For headless servers (no display), training works without `--render`. If you need pygame headless:
```bash
export SDL_VIDEODRIVER=dummy
```

### All platforms

**Python 3.9+** required. No ROS dependency. No GPU required (CPU training: ~2 min for 300 episodes).

**Tested with:**
- Python 3.12.3
- pymunk 7.2.0, gymnasium 1.2.3, shapely 2.1.2, numpy 2.4.2
- torch 2.11.0, matplotlib 3.10.8, wandb 0.25.1, imageio 2.37.2

### Verify installation

```bash
# Run 14 unit tests
python -m tests.test_sim

# Run deep audit of obs/action/reward correctness
python -m tests.audit_all
```

Expected: `14 passed, 0 failed` and `ALL CHECKS PASSED`.

### Setup Weights & Biases (optional)

```bash
pip install wandb
wandb login
# Paste your API key from https://wandb.ai/authorize
```

---

## Quick Start

```bash
# Train on curtain novelty with pygame visualization
python train_ppo.py --novelty curtain --episodes 300 --render

# Train on box novelty (headless, faster)
python train_ppo.py --novelty box --episodes 300

# Train on ball obstacle novelty
python train_ppo.py --novelty ball_obstacle --episodes 300

# Train with W&B logging (creates dashboard at wandb.ai)
python train_ppo.py --novelty curtain --episodes 300 --wandb

# Run all 3 experiments + generate learning curve plots
python run_all_experiments.py --episodes 300

# Evaluate a trained policy with rendering
python evaluate.py --novelty curtain --policy results/curtain_policy.pth --episodes 20 --render
```

After training, evaluation videos are automatically saved to `results/<novelty>_eval.mp4`.

---

## MDP Formulation

Each novelty scenario defines a **Semi-Markov Decision Process (SMDP)** instantiated when a symbolic operator fails during plan execution.

### State Space (S)

The observation is an **83-dimensional float vector**:

| Component | Dims | Range | Description |
|-----------|------|-------|-------------|
| Local occupancy grid | 64 (8×8) | {0, 0.5, 1.0} | Robot-centric, 0.1m/cell, aligned with heading. **0**=free, **0.5**=curtain (pushable), **1.0**=wall/rigid obstacle. |
| Forward obstruction | 1 | {0, 1} | 1.0 if obstacle within `ROBOT_RADIUS + 0.05m` directly ahead. |
| Relative poses | 10 (5×2) | ℝ | (Δx, Δy) meters from robot to waypoints: `bin`, `generic_object`, `table`, `atdoor`, `postdoor`. |
| Room encoding | 2 | {0, 1} | One-hot: `[room_1, room_2]`. Point-in-polygon test. |
| Holding | 1 | {0, 1} | 1.0 if robot holds `ball_1`. |
| Facing encoding | 5 | {0, 1} | One-hot: `[generic_object, table, bin_1, doorway_1, nothing]`. Angular proximity check. |

**Total: 83 floats.**

### Action Space (A)

| ID | Action | Effect |
|----|--------|--------|
| 0 | `move_forward` | Apply 0.5 m/s forward for 0.5s. Pymunk resolves all collisions. Robot moves ~0.19m (less if hitting obstacles). |
| 1 | `turn_left` | Rotate +0.175 rad (~10°) counter-clockwise. |
| 2 | `turn_right` | Rotate -0.175 rad (~10°) clockwise. |

### Transition Dynamics (T)

Transitions are **Pymunk 2D rigid-body physics**. Each `move_forward`:

1. Sets robot velocity to `(v·cos(θ), v·sin(θ))`
2. Steps Pymunk for 30 substeps at 60Hz
3. Pymunk resolves collisions: robot↔wall, robot↔box, robot↔curtain, robot↔ball
4. Zeroes velocity

Physics interactions per novelty:
- **Curtain**: `PivotJoint` + `DampedRotarySpring` (stiffness=15, damping=5). Swings open under sustained push, springs back.
- **Box**: Rigid body (2kg, friction=0.6). Slides when pushed. Friction decelerates it.
- **Obstacle ball**: Circle body (0.3kg, friction=0.3, elasticity=0.4). Rolls when pushed — different dynamics from box.

### Reward Function (R)

**Sparse reward:**

```
R(s, a, s') = +10.0   if target_predicates ⊆ current_predicates  (goal reached)
              -1.0    if symbolic action preconditions failed
               0.0    otherwise
```

The **target predicates** are derived from the failed operator's effects/preconditions via `compute_plannable_states()`. The **current predicates** are queried from the symbolic state each step. Success fires when every predicate in the target is satisfied — the current state may contain additional predicates.

| Scenario | Target predicates |
|----------|------------------|
| Curtain | `{(at, room_2, robot_1), (facing, nothing)}` |
| Box | `{(facing, bin_1)}` |
| Ball obstacle | `{(facing, bin_1)}` |

### Termination

- **`terminated = True`**: target predicates all satisfied (goal reached, +10 reward)
- **`truncated = True`**: `step_count >= max_steps`
- **Discount**: γ = 0.99

---

## Novelty Scenarios

### 1. Curtain Piercing

| Property | Value |
|----------|-------|
| Failed operator | `pass_through_door(room_1, room_2, doorway_1)` |
| Target predicates | `{(at, room_2, robot_1), (facing, nothing)}` |
| Robot start | (3.7, 1.5), heading=0° |
| Max steps | 60 |

**Physics**: Curtain body (0.05m × 0.6m, mass=0.5kg) hinged at top of doorway frame. `DampedRotarySpring` resists and pulls it back to vertical. Robot must apply repeated forward force to swing it open.

**What the agent learns**: Keep pushing forward at the doorway. The curtain yields after sustained contact.

### 2. Box Blocking Bin

| Property | Value |
|----------|-------|
| Failed operator | `approach(bin_1, room_2, nothing)` |
| Target predicates | `{(facing, bin_1)}` |
| Robot start | (4.5, 1.5), heading=0° |
| Max steps | 80 |

**Physics**: Box (0.3m × 0.3m, mass=2.0kg, friction=0.6) at (5.8, 1.5) blocks the direct path to the bin at (6.5, 1.5).

**What the agent learns**: Push the box sideways or past the bin, then navigate to face the bin.

### 3. Ball Blocking Bin

| Property | Value |
|----------|-------|
| Failed operator | `approach(bin_1, room_2, nothing)` |
| Target predicates | `{(facing, bin_1)}` |
| Robot start | (4.5, 1.5), heading=0° |
| Max steps | 80 |

**Physics**: Obstacle ball (radius=0.1m, mass=0.3kg, friction=0.3, elasticity=0.4) at (5.8, 1.5) blocks the bin. Unlike the box, it's round — it rolls more easily when pushed, requiring different approach angles.

**What the agent learns**: Push the ball at an angle so it rolls aside, then navigate to face the bin.

---

## Physics Parameters

All configurable via `WorldConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `linear_velocity` | 0.5 m/s | Robot forward speed |
| `angular_velocity` | 0.35 rad/s | Robot turn rate |
| `action_duration` | 0.5 s | Duration per primitive action |
| `floor_friction` | 0.8 | Robot-floor friction |
| `box_friction` | 0.6 | Box-floor friction |
| `curtain_spring_stiffness` | 15.0 N·m/rad | Curtain resistance |
| `curtain_spring_damping` | 5.0 N·m·s/rad | Curtain damping |
| `obstacle_ball_radius` | 0.1 m | Obstacle ball radius |
| `obstacle_ball_mass` | 0.3 kg | Obstacle ball mass |
| `ROBOT_RADIUS` | 0.17 m | LocoBot base radius |
| `ROBOT_MASS` | 5.0 kg | Robot mass |
| `BOX_MASS` | 2.0 kg | Box mass |

---

## Environment Layout

```
Y
3 ┌────────────────┬────────────────┐
  │                │                │
  │   Room 1       │    Room 2      │
  │                ║                │
1.5  [R]  [Obj]  ═╬═  [Obs]  [BIN]│
  │                ║                │
  │               [C]               │
0 └────────────────┴────────────────┘
  0       2        4       6        8  X

R   = Robot start (1.0, 1.5)
Obj = Task object (1.5, 1.5) — the object to pick up
C   = Curtain anchor (4.0, 1.9) — curtain novelty only
Obs = Obstacle position (5.8, 1.5) — box or ball novelty only
BIN = Bin (6.5, 1.5)
Doorway: x=4.0, y ∈ [1.1, 1.9]
```

---

## Project Structure

```
recyclebot_sim/
├── README.md                    # This file
├── requirements.txt             # Pinned dependencies
├── train_ppo.py                 # Train PPO on any novelty scenario
├── evaluate.py                  # Evaluate a trained policy
├── run_all_experiments.py       # Run all 3 experiments + plots
├── sim/
│   ├── __init__.py
│   ├── world.py                 # Pymunk physics world (SimWorld)
│   ├── env.py                   # Gymnasium environment (RecycleBotSimEnv)
│   └── renderer.py              # Pygame real-time renderer
├── tests/
│   ├── __init__.py
│   ├── test_sim.py              # 14-test validation suite
│   └── audit_all.py             # Deep audit of obs/action/reward
├── results/                     # Created by training scripts
│   ├── curtain_stats.csv        # Per-episode CSV logs
│   ├── curtain_policy.pth       # Saved PyTorch policy
│   ├── curtain_eval.mp4         # Evaluation video
│   └── ...
└── plots/                       # Created by run_all_experiments.py
    ├── learning_curves.png
    └── ...
```

---

## API Reference

### Creating a custom environment

```python
from sim import RecycleBotSimEnv, WorldConfig, NoveltyType

config = WorldConfig(
    novelty=NoveltyType.CURTAIN,
    robot_start=(3.7, 1.5),
    robot_start_heading=0.0,
    curtain_spring_stiffness=15.0,
)

env = RecycleBotSimEnv(
    world_config=config,
    target_predicates={("at", "room_2", "robot_1"), ("facing", "nothing")},
    max_steps=60,
    render_mode="human",  # "human" for pygame, "rgb_array" for video
)

obs, info = env.reset()
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

### Accessing the physics world directly

```python
world = env.world

# Symbolic queries (same interface as ROS services on real robot)
world.query_at("robot_1", "room_1")      # bool
world.query_facing("bin_1")               # bool
world.query_hold("ball_1")               # bool
world.query_contain("ball_1", "bin_1")   # bool

# Sub-symbolic state
grid = world.get_local_occupancy_grid(size=8, resolution=0.1)  # (8,8) ndarray
rx, ry = world.get_relative_pose("bin")   # (float, float)
pos = world.get_robot_position()           # (x, y)
heading = world.get_robot_heading()        # radians

# Direct control
world.move_forward()
world.turn_left()
world.turn_right()
```

---

## W&B Integration

Training with `--wandb` logs to a `recyclebot-sim` project:

```bash
python train_ppo.py --novelty curtain --episodes 300 --wandb
```

Logged metrics per episode: `reward`, `steps`, `success`, `success_rate`, `avg_reward_20`, `loss`. Evaluation videos are uploaded as `wandb.Video` artifacts at the end of training.

---

## Tuning Tips

- **Curtain too hard**: Decrease `curtain_spring_stiffness` (try 5–15)
- **Box doesn't move**: Decrease `box_friction` (try 0.2–0.4) or increase `ROBOT_MASS`
- **Ball rolls too far**: Increase obstacle ball friction, decrease elasticity
- **Not exploring**: Increase entropy coefficient in PPO (default 0.01, try 0.05)
- **Sparse reward too hard**: The `_get_distance_to_goal()` method is implemented and available — switch to dense reward by uncommenting distance shaping in `_compute_reward()`
