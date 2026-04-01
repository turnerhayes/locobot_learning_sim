"""
Run all three novelty experiments and produce learning curve plots.

Usage:
  python run_all_experiments.py
  python run_all_experiments.py --episodes 500
  python run_all_experiments.py --episodes 100 --render
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from train_ppo import train, SCENARIOS


def plot_learning_curves(save_dir: str, plot_dir: str, window: int = 20):
    """Plot learning curves from saved stats CSVs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("RecycleBot Novelty Learning Curves", fontsize=14, fontweight="bold")

    scenarios = ["curtain", "box", "ball_obstacle"]
    colors = ["#534AB7", "#1D9E75", "#D85A30"]

    for idx, (scenario, color) in enumerate(zip(scenarios, colors)):
        stats_file = os.path.join(save_dir, f"{scenario}_stats.csv")
        if not os.path.exists(stats_file):
            axes[idx].set_title(f"{scenario} (no data)")
            continue

        data = np.genfromtxt(stats_file, delimiter=",", skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        episodes = data[:, 0].astype(int)
        rewards = data[:, 2]
        successes = data[:, 3].astype(int)

        # Smoothed reward
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        # Smoothed success rate
        success_rate = np.convolve(successes, np.ones(window) / window, mode="valid") * 100

        ax = axes[idx]
        ax2 = ax.twinx()

        ax.plot(episodes[:len(smoothed)], smoothed, color=color, linewidth=1.5, label="Reward")
        ax.fill_between(episodes[:len(smoothed)], smoothed, alpha=0.15, color=color)
        ax2.plot(episodes[:len(success_rate)], success_rate, color=color, linewidth=1.5,
                 linestyle="--", alpha=0.7, label="Success %")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward (smoothed)")
        ax2.set_ylabel("Success rate (%)")
        ax2.set_ylim(0, 105)
        ax.set_title(f"{scenario.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nLearning curves saved to: {plot_path}")

    # Also save individual plots
    for idx, (scenario, color) in enumerate(zip(scenarios, colors)):
        stats_file = os.path.join(save_dir, f"{scenario}_stats.csv")
        if not os.path.exists(stats_file):
            continue

        data = np.genfromtxt(stats_file, delimiter=",", skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        episodes = data[:, 0].astype(int)
        steps = data[:, 1].astype(int)
        rewards = data[:, 2]
        successes = data[:, 3].astype(int)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle(f"{scenario.replace('_', ' ').title()} Novelty — Training Details", fontsize=13)

        # Reward
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(episodes, rewards, alpha=0.3, color=color, linewidth=0.5)
        ax1.plot(episodes[:len(smoothed)], smoothed, color=color, linewidth=2)
        ax1.set_ylabel("Episode Reward")
        ax1.grid(True, alpha=0.3)

        # Steps
        smoothed_steps = np.convolve(steps.astype(float), np.ones(window) / window, mode="valid")
        ax2.plot(episodes, steps, alpha=0.3, color=color, linewidth=0.5)
        ax2.plot(episodes[:len(smoothed_steps)], smoothed_steps, color=color, linewidth=2)
        ax2.set_ylabel("Steps per Episode")
        ax2.grid(True, alpha=0.3)

        # Success rate
        success_rate = np.convolve(successes, np.ones(window) / window, mode="valid") * 100
        ax3.plot(episodes[:len(success_rate)], success_rate, color=color, linewidth=2)
        ax3.fill_between(episodes[:len(success_rate)], success_rate, alpha=0.15, color=color)
        ax3.set_ylabel("Success Rate (%)")
        ax3.set_ylim(0, 105)
        ax3.set_xlabel("Episode")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        individual_path = os.path.join(plot_dir, f"{scenario}_details.png")
        plt.savefig(individual_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {scenario} details saved to: {individual_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all RecycleBot novelty experiments")
    parser.add_argument("--episodes", type=int, default=300, help="Episodes per experiment")
    parser.add_argument("--render", action="store_true", help="Enable pygame rendering")
    parser.add_argument("--save-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Plot output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("RecycleBot Simulation — Full Experiment Suite")
    print(f"Episodes per scenario: {args.episodes}")
    print("=" * 70)

    results = {}
    for scenario_name in ["curtain", "box", "ball_obstacle"]:
        successes, total = train(scenario_name, args.episodes, args.render, args.save_dir)
        results[scenario_name] = (successes, total)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Novelty':<15} {'Successes':>10} {'Total':>8} {'Rate':>8}")
    print("-" * 45)
    for name, (s, t) in results.items():
        rate = s / t * 100 if t > 0 else 0
        print(f"{name:<15} {s:>10} {t:>8} {rate:>7.1f}%")
    print("=" * 70)

    # Plot
    plot_learning_curves(args.save_dir, args.plot_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
