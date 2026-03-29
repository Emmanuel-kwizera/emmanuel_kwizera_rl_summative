"""
main.py — AfAlm Drone Crop Disease Management
----------------------------------------------
Entry point for running and visualising the environment.

Usage:
    # Random agent (no model — just shows the environment)
    python main.py --mode random

    # Run best saved model
    python main.py --mode best

    # Run specific algorithm
    python main.py --mode dqn
    python main.py --mode ppo
"""

import argparse
import os
import time
import numpy as np

from environment.custom_env import CropDroneEnv


def run_random(env: CropDroneEnv, episodes: int = 3):
    """Demonstrate environment with a random agent (no training)."""
    print("\n" + "="*55)
    print("  AfAlm — Random Agent Demo (no model)")
    print("="*55)
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        print(f"\n  Episode {ep + 1}")
        print(f"  {'─'*40}")
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            print(
                f"  Step {info['step']:>3d} | "
                f"Action: {info['last_action']:<20s} | "
                f"Reward: {reward:>7.1f} | "
                f"Fuel: {info['fuel']:>3d} | "
                f"Infected: {info['infected_count']:>2d} | "
                f"{info['last_event']}"
            )
            time.sleep(0.05)
        print(f"\n  Episode {ep+1} finished — Total reward: {total_reward:.1f}")
        print(f"  Treated: {info['treated_count']}  Dead: {info['dead_count']}")
    env.close()


def run_model(env: CropDroneEnv, algo: str):
    """Run a saved Stable-Baselines3 model."""
    from stable_baselines3 import DQN, PPO

    model_map = {
        "dqn": (DQN,  "models/dqn/best_model"),
        "ppo": (PPO,  "models/pg/ppo_best"),
    }

    if algo not in model_map:
        print(f"Unknown algorithm: {algo}. Choose from: dqn, ppo")
        return

    ModelClass, model_path = model_map[algo]

    # Try best_model, then fall back to latest checkpoint
    path = model_path + ".zip"
    if not os.path.exists(path):
        print(f"  Model not found at {path}. Have you trained it?")
        print(f"  Run: python training/{algo}_training.py")
        return

    print(f"\n  Loading {algo.upper()} model from {path}")
    model = ModelClass.load(model_path, env=env)

    print("\n" + "="*55)
    print(f"  AfAlm — {algo.upper()} Agent Demo")
    print("="*55)

    obs, info = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        done = terminated or truncated
        print(
            f"  Step {info['step']:>3d} | "
            f"Action: {info['last_action']:<20s} | "
            f"Reward: {reward:>7.1f} | "
            f"Fuel: {info['fuel']:>3d} | "
            f"Infected: {info['infected_count']:>2d} | "
            f"{info['last_event']}"
        )

    print(f"\n  Episode finished — Total reward: {total_reward:.1f}")
    print(f"  Treated: {info['treated_count']}  Dead: {info['dead_count']}")
    env.close()


def find_best_model():
    """Scan models/ folder and find the best performing saved model."""
    candidates = [
        ("dqn", "models/dqn/best_model.zip"),
        ("ppo", "models/pg/ppo_best.zip"),
    ]
    for algo, path in candidates:
        if os.path.exists(path):
            return algo
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AfAlm Drone RL Environment Runner")
    parser.add_argument(
        "--mode",
        choices=["random", "best", "dqn", "ppo"],
        default="random",
        help="Agent mode: random (no model), best (auto-detect), or specific algo",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to run (random mode only)"
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Disable Pygame rendering (terminal output only)"
    )
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"

    env = CropDroneEnv(
        grid_size=8,
        max_steps=200,
        max_fuel=100,
        max_payload=3,
        spread_interval=10,
        spread_prob=0.20,
        n_infected=5,
        render_mode=render_mode,
    )

    if args.mode == "random":
        run_random(env, episodes=args.episodes)
    elif args.mode == "best":
        best = find_best_model()
        if best is None:
            print("  No trained models found — running random agent instead.")
            run_random(env, episodes=args.episodes)
        else:
            print(f"  Found best model: {best.upper()}")
            run_model(env, best)
    else:
        run_model(env, args.mode)
