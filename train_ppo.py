"""
Train PPO on RecycleBot novelty scenarios.

Usage:
  python train_ppo.py --novelty curtain --episodes 300 --render
  python train_ppo.py --novelty box --episodes 300
  python train_ppo.py --novelty ball_obstacle --episodes 300
  python train_ppo.py --novelty curtain --episodes 300 --wandb

Each novelty scenario mirrors the real robot's HybridAgent flow:
  1. Planner generates a plan for the recycle task
  2. An operator fails due to novelty
  3. compute_plannable_states() determines the recovery target
  4. PPO learns a primitive policy to reach the plannable state
"""

import argparse
import math
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from sim.world import WorldConfig, NoveltyType
from sim.env import RecycleBotSimEnv

# ============================================================
# Novelty Scenario Definitions
# ============================================================

SCENARIOS = {
    "curtain": {
        "description": "Curtain blocks doorway. Robot must push through to reach room_2.",
        "novelty": NoveltyType.CURTAIN,
        "failed_operator_name": "pass_through_door",
        "failed_operator_params": ["room_1", "room_2", "doorway_1"],
        "target_predicates": {("at", "room_2", "robot_1"), ("facing", "nothing")},
        "robot_start": (3.7, 1.5),
        "robot_start_heading": 0.0,
        "max_steps": 60,
    },
    "box": {
        "description": "Box blocks bin. Robot must push box aside to face the bin.",
        "novelty": NoveltyType.BOX,
        "failed_operator_name": "approach",
        "failed_operator_params": ["bin_1", "room_2", "nothing"],
        "target_predicates": {("facing", "bin_1")},
        "robot_start": (4.5, 1.5),
        "robot_start_heading": 0.0,
        "max_steps": 80,
    },
    "ball_obstacle": {
        "description": "Ball blocks bin. Robot must push ball aside to face the bin.",
        "novelty": NoveltyType.BALL_OBSTACLE,
        "failed_operator_name": "approach",
        "failed_operator_params": ["bin_1", "room_2", "nothing"],
        "target_predicates": {("facing", "bin_1")},
        "robot_start": (4.5, 1.5),
        "robot_start_heading": 0.0,
        "max_steps": 80,
    },
}


# ============================================================
# PPO Implementation (standalone, no ROS dependency)
# ============================================================

import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def act(self, state):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def evaluate(self, states, actions):
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states).squeeze(-1)
        return log_probs, values, entropy


class PPOTrainer:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_dones = []

    def select_action(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, logprob = self.policy.act(state_t)
        return action, logprob

    def store(self, state, action, logprob, reward, done):
        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_logprobs.append(logprob)
        self.buffer_rewards.append(reward)
        self.buffer_dones.append(done)

    def update(self):
        returns = []
        discounted = 0
        for r, d in zip(reversed(self.buffer_rewards), reversed(self.buffer_dones)):
            if d:
                discounted = 0
            discounted = r + self.gamma * discounted
            returns.insert(0, discounted)

        returns = torch.FloatTensor(returns).to(device)
        if returns.std() > 1e-6:
            returns = (returns - returns.mean()) / returns.std()

        old_states = torch.FloatTensor(np.array(self.buffer_states)).to(device)
        old_actions = torch.LongTensor(self.buffer_actions).to(device)
        old_logprobs = torch.FloatTensor(self.buffer_logprobs).to(device)

        total_loss = 0
        for _ in range(self.k_epochs):
            logprobs, values, entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = returns - values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * (values - returns) ** 2 - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.mean().item()

        self.buffer_states.clear()
        self.buffer_actions.clear()
        self.buffer_logprobs.clear()
        self.buffer_rewards.clear()
        self.buffer_dones.clear()

        return total_loss / self.k_epochs

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=device))


# ============================================================
# Video Recording
# ============================================================

def record_eval_video(ppo, scenario, video_path, num_episodes=3):
    """Record evaluation episodes as an mp4 video."""
    try:
        import imageio
    except ImportError:
        print("Warning: imageio not installed, skipping video recording. pip install imageio imageio-ffmpeg")
        return None

    config = WorldConfig(
        novelty=scenario["novelty"],
        robot_start=scenario["robot_start"],
        robot_start_heading=scenario["robot_start_heading"],
    )
    vid_env = RecycleBotSimEnv(
        novelty=scenario["novelty"],
        failed_operator_name=scenario["failed_operator_name"],
        failed_operator_params=scenario["failed_operator_params"],
        target_predicates=scenario["target_predicates"],
        max_steps=scenario["max_steps"],
        render_mode="rgb_array",
        world_config=config,
    )

    frames = []
    for ep in range(num_episodes):
        obs, _ = vid_env.reset()
        done = False
        step = 0
        while not done and step < scenario["max_steps"]:
            with torch.no_grad():
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                logits = ppo.policy.actor(state_t)
                action = logits.argmax(dim=-1).item()

            obs, reward, terminated, truncated, info = vid_env.step(action)
            done = terminated or truncated
            step += 1

            frame = vid_env.render()
            if frame is not None:
                frames.append(frame)

    vid_env.close()

    if frames:
        os.makedirs(os.path.dirname(video_path) if os.path.dirname(video_path) else ".", exist_ok=True)
        imageio.mimsave(video_path, frames, fps=15)
        print(f"  Video saved: {video_path} ({len(frames)} frames)")
        return video_path
    return None


# ============================================================
# Evaluation
# ============================================================

def run_eval(ppo, scenario, num_eval_episodes=20):
    """Run deterministic evaluation rollouts. Returns (success_rate, avg_reward, avg_steps)."""
    config = WorldConfig(
        novelty=scenario["novelty"],
        robot_start=scenario["robot_start"],
        robot_start_heading=scenario["robot_start_heading"],
    )
    eval_env = RecycleBotSimEnv(
        novelty=scenario["novelty"],
        failed_operator_name=scenario["failed_operator_name"],
        failed_operator_params=scenario["failed_operator_params"],
        target_predicates=scenario["target_predicates"],
        max_steps=scenario["max_steps"],
        world_config=config,
    )

    successes = 0
    total_reward = 0.0
    total_steps = 0

    for ep in range(num_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done and ep_steps < scenario["max_steps"]:
            with torch.no_grad():
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                logits = ppo.policy.actor(state_t)
                action = logits.argmax(dim=-1).item()

            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

        if terminated:
            successes += 1
        total_reward += ep_reward
        total_steps += ep_steps

    eval_env.close()

    success_rate = successes / num_eval_episodes
    avg_reward = total_reward / num_eval_episodes
    avg_steps = total_steps / num_eval_episodes
    return success_rate, avg_reward, avg_steps


# ============================================================
# Training Loop
# ============================================================

def train(scenario_name: str, num_episodes: int, render: bool = False,
          save_dir: str = "results", use_wandb: bool = False,
          eval_every: int = 50, num_eval_episodes: int = 20,
          target_success_rate: float = 0.80):
    scenario = SCENARIOS[scenario_name]

    print(f"\n{'='*60}")
    print(f"Training: {scenario_name}")
    print(f"Description: {scenario['description']}")
    print(f"Failed operator: {scenario['failed_operator_name']} {scenario['failed_operator_params']}")
    print(f"Target predicates: {scenario['target_predicates']}")
    print(f"Episodes: {num_episodes}, Max steps: {scenario['max_steps']}")
    print(f"Eval every {eval_every} eps, {num_eval_episodes} rollouts, target {target_success_rate*100:.0f}%")
    print(f"{'='*60}\n")

    # W&B init
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project="recyclebot-sim",
            name=f"{scenario_name}_{num_episodes}ep",
            config={
                "scenario": scenario_name,
                "num_episodes": num_episodes,
                "max_steps": scenario["max_steps"],
                "failed_operator": scenario["failed_operator_name"],
                "target_predicates": str(scenario["target_predicates"]),
                "lr": 3e-4,
                "gamma": 0.99,
                "eps_clip": 0.2,
                "k_epochs": 4,
                "reward": "sparse",
                "eval_every": eval_every,
                "num_eval_episodes": num_eval_episodes,
                "target_success_rate": target_success_rate,
            },
        )

    # Build environment
    world_config = WorldConfig(
        novelty=scenario["novelty"],
        robot_start=scenario["robot_start"],
        robot_start_heading=scenario["robot_start_heading"],
    )

    env = RecycleBotSimEnv(
        novelty=scenario["novelty"],
        failed_operator_name=scenario["failed_operator_name"],
        failed_operator_params=scenario["failed_operator_params"],
        target_predicates=scenario["target_predicates"],
        max_steps=scenario["max_steps"],
        render_mode="human" if render else None,
        world_config=world_config,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Actions: {[a['name'] for a in env.action_list]}")

    ppo = PPOTrainer(state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4)

    # Renderer
    renderer = None
    if render:
        try:
            from sim.renderer import PygameRenderer
            renderer = PygameRenderer(env.world, show_grid=False)
        except ImportError:
            print("Warning: pygame not available")
            render = False

    # Stats file
    os.makedirs(save_dir, exist_ok=True)
    stats_file = os.path.join(save_dir, f"{scenario_name}_stats.csv")
    eval_stats_file = os.path.join(save_dir, f"{scenario_name}_eval_stats.csv")
    with open(stats_file, "w") as f:
        f.write("episode,steps,reward,success,avg_loss\n")
    with open(eval_stats_file, "w") as f:
        f.write("episode,eval_success_rate,eval_avg_reward,eval_avg_steps\n")

    train_successes = 0
    reward_history = []
    reached_target = False

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done and episode_steps < scenario["max_steps"]:
            action, logprob = ppo.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            ppo.store(obs, action, logprob, reward, done)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1

            if render and renderer is not None:
                act_name = env.action_list[action]["name"]
                renderer.draw(
                    step=episode_steps, episode=episode,
                    reward=episode_reward, info=f"Act:{act_name}",
                )

        avg_loss = ppo.update()
        success = terminated

        if success:
            train_successes += 1

        reward_history.append(episode_reward)
        recent_avg = np.mean(reward_history[-20:])

        with open(stats_file, "a") as f:
            f.write(f"{episode},{episode_steps},{episode_reward:.4f},{int(success)},{avg_loss:.4f}\n")

        if wandb_run is not None:
            import wandb
            wandb.log({
                "episode": episode,
                "train/reward": episode_reward,
                "train/steps": episode_steps,
                "train/success": int(success),
                "train/success_rate": train_successes / (episode + 1),
                "train/avg_reward_20": recent_avg,
                "train/loss": avg_loss,
            })

        if episode % 10 == 0 or success:
            success_rate = train_successes / (episode + 1) * 100
            print(f"  Ep {episode:4d} | Steps: {episode_steps:3d} | Reward: {episode_reward:7.2f} | "
                  f"Success: {'YES' if success else 'no ':3s} | Rate: {success_rate:5.1f}% | "
                  f"AvgR(20): {recent_avg:7.2f} | Loss: {avg_loss:.4f}")

        # ----- Periodic evaluation -----
        if (episode + 1) % eval_every == 0:
            print(f"\n  --- Eval at episode {episode + 1} ({num_eval_episodes} rollouts) ---")
            eval_sr, eval_reward, eval_steps = run_eval(ppo, scenario, num_eval_episodes)
            print(f"  Eval success rate: {eval_sr*100:.1f}%  |  Avg reward: {eval_reward:.2f}  |  Avg steps: {eval_steps:.1f}")

            with open(eval_stats_file, "a") as f:
                f.write(f"{episode + 1},{eval_sr:.4f},{eval_reward:.4f},{eval_steps:.1f}\n")

            if wandb_run is not None:
                import wandb
                log_data = {
                    "eval/success_rate": eval_sr,
                    "eval/avg_reward": eval_reward,
                    "eval/avg_steps": eval_steps,
                }
                # Record eval video and log to W&B
                video_path = os.path.join(save_dir, f"{scenario_name}_eval_ep{episode+1}.mp4")
                vid_file = record_eval_video(ppo, scenario, video_path, num_episodes=3)
                if vid_file and os.path.exists(vid_file):
                    log_data["eval/video"] = wandb.Video(vid_file, fps=15, format="mp4")
                wandb.log(log_data)

            if eval_sr >= target_success_rate:
                print(f"\n  *** Target success rate {target_success_rate*100:.0f}% reached at episode {episode+1}! ***\n")
                reached_target = True
                break

    # Save final policy
    policy_path = os.path.join(save_dir, f"{scenario_name}_policy.pth")
    ppo.save(policy_path)

    # Final eval video
    video_path = os.path.join(save_dir, f"{scenario_name}_eval_final.mp4")
    vid_file = record_eval_video(ppo, scenario, video_path, num_episodes=5)

    if wandb_run is not None and vid_file and os.path.exists(vid_file):
        import wandb
        wandb.log({"eval/final_video": wandb.Video(vid_file, fps=15, format="mp4")})

    # Summary
    final_ep = episode + 1
    train_sr = train_successes / final_ep * 100
    avg_reward = np.mean(reward_history)
    print(f"\n{'='*60}")
    print(f"Training Complete: {scenario_name}")
    print(f"  Episodes trained: {final_ep}")
    print(f"  Train success rate: {train_successes}/{final_ep} ({train_sr:.1f}%)")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Reached target: {reached_target}")
    print(f"  Policy saved: {policy_path}")
    print(f"  Stats: {stats_file}")
    print(f"  Eval stats: {eval_stats_file}")
    print(f"{'='*60}")

    env.close()
    if renderer is not None:
        renderer.close()
    if wandb_run is not None:
        import wandb
        wandb.finish()

    return train_successes, final_ep


def main():
    parser = argparse.ArgumentParser(description="Train PPO on RecycleBot novelty scenarios")
    parser.add_argument("--novelty", type=str, required=True,
                        choices=list(SCENARIOS.keys()),
                        help="Which novelty scenario to train on")
    parser.add_argument("--episodes", type=int, default=300,
                        help="Number of training episodes")
    parser.add_argument("--render", action="store_true",
                        help="Enable pygame rendering during training")
    parser.add_argument("--save-dir", type=str, default="results",
                        help="Directory to save policies and stats")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Run evaluation every N training episodes")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Number of eval rollouts per checkpoint")
    parser.add_argument("--target-sr", type=float, default=0.80,
                        help="Stop training when eval success rate exceeds this (0-1)")
    args = parser.parse_args()

    train(args.novelty, args.episodes, args.render, args.save_dir, args.wandb,
          args.eval_every, args.eval_episodes, args.target_sr)


if __name__ == "__main__":
    main()
