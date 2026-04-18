"""
Evaluate a trained PPO policy on a novelty scenario.

Usage:
  python evaluate.py --novelty curtain --policy results/curtain_policy.pth --episodes 50
  python evaluate.py --novelty curtain --policy results/curtain_policy.pth --episodes 10 --render
  python evaluate.py --novelty box --policy results/box_policy.pth --episodes 50 --render
"""

from typing import cast, Tuple
import argparse
import os
import sys
import time
from gymnasium.spaces import Discrete
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from sim.world import WorldConfig
from sim.env import RecycleBotSimEnv
from train_ppo import SCENARIOS, PPOTrainer


def evaluate(
    scenario_name: str,
    policy_path: str,
    num_episodes: int = 50,
    render: bool = False,
    deterministic: bool = True,
    render_delay: float = 0.05,
):
    scenario = SCENARIOS[scenario_name]

    print(f"\n{'='*60}")
    print(f"Evaluating: {scenario_name}")
    print(f"Policy: {policy_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Deterministic: {deterministic}")
    print(f"{'='*60}\n")

    # Build environment
    world_config = WorldConfig(
        novelty=scenario["novelty"],
        robot_start=scenario["robot_start"],
        robot_start_heading=scenario["robot_start_heading"],
    )

    include_symbolic = scenario.get("include_symbolic_actions", False)

    env = RecycleBotSimEnv(
        novelty=scenario["novelty"],
        failed_operator_name=scenario["failed_operator_name"],
        failed_operator_params=scenario["failed_operator_params"],
        target_predicates=scenario["target_predicates"],
        include_symbolic_actions=include_symbolic,
        max_steps=scenario["max_steps"],
        render_mode="human" if render else None,
        world_config=world_config,
    )

    state_dim = cast(Tuple[int, ...], env.observation_space.shape)[0]
    action_space = cast(Discrete, env.action_space)
    action_dim = action_space.n

    # Load policy
    ppo = PPOTrainer(state_dim, action_dim)
    ppo.load(policy_path)
    print(f"Loaded policy from {policy_path}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Setup renderer
    renderer = None
    if render:
        try:
            from sim.renderer import PygameRenderer
            renderer = PygameRenderer(env.world, show_grid=True)
        except ImportError:
            print("Warning: pygame not available, skipping rendering")
            render = False

    # Evaluate
    successes = 0
    rewards_all = []
    steps_all = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done and episode_steps < scenario["max_steps"]:
            if deterministic:
                # Use argmax for deterministic evaluation
                import torch
                with torch.no_grad():
                    state_t = torch.FloatTensor(obs).unsqueeze(0)
                    logits = ppo.policy.actor(state_t)
                    action = logits.argmax(dim=-1).item()
            else:
                action, _ = ppo.select_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1

            if render and renderer is not None:
                act_name = env.action_list[action]["name"]
                renderer.draw(
                    step=episode_steps,
                    episode=episode,
                    reward=episode_reward,
                    info=f"Act:{act_name}",
                )
                time.sleep(render_delay)

        success = terminated
        if success:
            successes += 1
        rewards_all.append(episode_reward)
        steps_all.append(episode_steps)

        marker = "✓" if success else "✗"
        print(f"  Ep {episode:3d} | {marker} | Steps: {episode_steps:3d} | Reward: {episode_reward:7.2f}")

    # Summary
    success_rate = successes / num_episodes * 100
    avg_reward = np.mean(rewards_all)
    std_reward = np.std(rewards_all)
    avg_steps = np.mean(steps_all)

    print(f"\n{'='*60}")
    print(f"Evaluation Results: {scenario_name}")
    print(f"  Success rate:  {successes}/{num_episodes} ({success_rate:.1f}%)")
    print(f"  Avg reward:    {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Avg steps:     {avg_steps:.1f}")
    print(f"{'='*60}")

    env.close()
    if renderer is not None:
        renderer.close()

    return success_rate, avg_reward


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy")
    parser.add_argument("--novelty", type=str, required=True,
                        choices=list(SCENARIOS.keys()),
                        help="Novelty scenario to evaluate")
    parser.add_argument("--policy", type=str, required=True,
                        help="Path to trained policy .pth file")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Enable pygame rendering")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic (sampled) actions instead of deterministic")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Render delay per step in seconds (only with --render)")
    args = parser.parse_args()

    evaluate(
        args.novelty,
        args.policy,
        args.episodes,
        args.render,
        deterministic=not args.stochastic,
        render_delay=args.delay,
    )


if __name__ == "__main__":
    main()
