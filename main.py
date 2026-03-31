"""
main.py — AfAlm Drone Crop Disease Management
----------------------------------------------
Entry point for running and visualising the environment.

Usage:
    python main.py --mode random          # random agent demo
    python main.py --mode random --episodes 3
    python main.py --mode best            # auto-detect best trained model
    python main.py --mode dqn             # original DQN
    python main.py --mode ppo             # original PPO
    python main.py --mode ppo_v2          # improved PPO v2 (VecNormalize)
    python main.py --mode reinforce       # REINFORCE
    python main.py --mode random --no-render
"""

import argparse
import os
import time
import numpy as np

from environment.custom_env import CropDroneEnv


def _pump(env):
    """Pump pygame events. Returns False if user closed window."""
    try:
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return False
    except Exception:
        pass
    return True


def run_random(env: CropDroneEnv, episodes: int = 1):
    print("\n" + "="*62)
    print("  AfAlm — Random Agent Demo")
    print("="*62)
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward, done = 0.0, False
        print(f"\n  Episode {ep + 1}  {'─'*45}")
        while not done:
            if not _pump(env): return
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            print(f"  Step {info['step']:>3d} | {info['last_action']:<20s} | "
                  f"Reward: {reward:>7.1f} | Fuel: {info['fuel']:>3d} | "
                  f"Infected: {info['infected_count']:>2d} | {info['last_event']}")
            time.sleep(0.15)
        print(f"\n  Episode {ep+1} — reward={total_reward:.1f}  "
              f"treated={info['treated_count']}  dead={info['dead_count']}")
    env.close()


def run_sb3_model(env: CropDroneEnv, algo: str):
    """DQN or original PPO — no VecNormalize needed."""
    from stable_baselines3 import DQN, PPO

    search = {
        "dqn": (DQN, ["models/improved/dqn/best_model", "models/dqn/best_model"]),
        "ppo": (PPO, ["models/pg/ppo_best"]),
    }
    if algo not in search:
        print(f"  Unknown algo: {algo}"); env.close(); return

    ModelClass, paths = search[algo]
    model_path = next((p for p in paths if os.path.exists(p + ".zip")), None)
    if model_path is None:
        print(f"  No {algo.upper()} model found — train it first."); env.close(); return

    print(f"\n  Loading {algo.upper()} → {model_path}.zip")
    model = ModelClass.load(model_path, env=env)

    print("\n" + "="*62 + f"\n  AfAlm — {algo.upper()} Agent\n" + "="*62)
    obs, info = env.reset()
    total_reward, done = 0.0, False
    while not done:
        if not _pump(env): return
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        done = terminated or truncated
        print(f"  Step {info['step']:>3d} | {info['last_action']:<20s} | "
              f"Reward: {reward:>7.1f} | Fuel: {info['fuel']:>3d} | "
              f"Infected: {info['infected_count']:>2d} | {info['last_event']}")
        time.sleep(0.15)
    print(f"\n  Finished — reward={total_reward:.1f}  "
          f"treated={info['treated_count']}  dead={info['dead_count']}")
    env.close()


def run_ppo_v2(render_mode: str):
    """
    PPO v2 needs the EXACT same env stack it was trained on:
    CropDroneEnv (easy) → RewardShapingWrapper → Monitor → VecNormalize
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from training.improved.improved_ppo_v2 import (
        RewardShapingWrapper, EASY_ENV_KWARGS
    )

    # Find model — check global best first, then per-run bests
    model_candidates = [
        "models/improved/ppo_v2/best_model",
        "models/improved/ppo_v2/run_02/best_model",
        "models/improved/ppo_v2/run_01/best_model",
        "models/improved/ppo_v2/run_05/best_model",
        "models/improved/ppo_v2/run_03/best_model",
        "models/improved/ppo_v2/run_04/best_model",
    ]
    model_path = next((p for p in model_candidates if os.path.exists(p+".zip")), None)
    if model_path is None:
        print("  PPO v2 model not found.")
        print("  Train: python training/improved/improved_ppo_v2.py"); return

    # Find VecNormalize stats
    norm_candidates = [
        "models/improved/ppo_v2/vecnormalize_best.pkl",
        os.path.join(os.path.dirname(model_path), "vecnormalize.pkl"),
        os.path.join(os.path.dirname(model_path), "vecnormalize_best.pkl"),
    ]
    vecnorm_path = next((p for p in norm_candidates if os.path.exists(p)), None)

    print(f"\n  Loading PPO v2 → {model_path}.zip")
    print(f"  VecNormalize → {vecnorm_path or 'NOT FOUND (using fresh stats)'}")

    # Reconstruct training env stack
    def make_env():
        base = CropDroneEnv(**EASY_ENV_KWARGS, render_mode=render_mode)
        return Monitor(RewardShapingWrapper(base, proximity_scale=2.0))

    vec_env = DummyVecEnv([make_env])

    if vecnorm_path:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                               clip_obs=10.0, training=False)

    model = PPO.load(model_path, env=vec_env)

    # Get the underlying CropDroneEnv for direct rendering access
    inner = vec_env.envs[0]               # Monitor
    base_env = inner.env.env              # RewardShaping → CropDroneEnv

    print("\n" + "="*62 + "\n  AfAlm — PPO v2 Agent (improved)\n" + "="*62)

    obs = vec_env.reset()
    total_reward, done, step = 0.0, False, 0

    while not done:
        if not _pump(base_env): break

        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        done = bool(dones[0])
        step += 1
        r = float(rewards[0])
        total_reward += r

        # Trigger rendering manually (VecEnv bypasses env.step render call)
        if render_mode == "human":
            base_env.render_mode = "human"
            base_env.render()

        infected = int(np.sum(
            (base_env.grid == 1) | (base_env.grid == 2) | (base_env.grid == 3)
        ))
        treated  = int(np.sum(base_env.grid == 4))
        print(f"  Step {step:>3d} | Fuel: {base_env.fuel:>3d} | "
              f"Infected: {infected:>2d} | Treated: {treated:>2d} | "
              f"Reward: {r:>7.1f}")
        time.sleep(0.15)

    print(f"\n  Finished — reward={total_reward:.1f}")
    vec_env.close()


def run_reinforce_model(env: CropDroneEnv):
    import torch
    import torch.nn as nn

    candidates = [
        ("models/improved/pg/reinforce_best.pt", 128),
        ("models/pg/reinforce_best.pt",           256),
    ]
    loaded = None
    for path, hidden in candidates:
        if not os.path.exists(path): continue
        obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden),  nn.ReLU(),
                    nn.Linear(hidden, n_actions),
                )
            def forward(self, x):
                return torch.softmax(self.net(x), dim=-1)
        try:
            policy = Net()
            policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            policy.eval()
            loaded = (policy, path)
            print(f"\n  Loading REINFORCE → {path}  (hidden={hidden})")
            break
        except Exception:
            continue

    if loaded is None:
        print("  REINFORCE model not found."); env.close(); return

    policy, _ = loaded
    print("\n" + "="*62 + "\n  AfAlm — REINFORCE Agent\n" + "="*62)
    obs, info = env.reset()
    total_reward, done = 0.0, False
    while not done:
        if not _pump(env): return
        with torch.no_grad():
            probs  = policy(torch.FloatTensor(obs).unsqueeze(0))
            action = probs.argmax(dim=-1).item()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        print(f"  Step {info['step']:>3d} | {info['last_action']:<20s} | "
              f"Reward: {reward:>7.1f} | Fuel: {info['fuel']:>3d} | "
              f"Infected: {info['infected_count']:>2d} | {info['last_event']}")
        time.sleep(0.15)
    print(f"\n  Finished — reward={total_reward:.1f}  "
          f"treated={info['treated_count']}  dead={info['dead_count']}")
    env.close()


def find_best_model():
    """Scan all model folders and return the algo key with highest reward."""
    candidates = {
        "ppo_v2":     ("models/improved/ppo_v2/best_model.zip",
                       "models/improved/ppo_v2/best_reward.txt"),
        "dqn":        ("models/improved/dqn/best_model.zip",
                       "models/improved/dqn/best_reward.txt"),
        "reinforce":  ("models/improved/pg/reinforce_best.pt",
                       "models/improved/pg/reinforce_best_reward.txt"),
        "ppo_orig":   ("models/pg/ppo_best.zip",
                       "models/pg/ppo_best_reward.txt"),
        "reinforce_orig": ("models/pg/reinforce_best.pt",
                           "models/pg/reinforce_best_reward.txt"),
        "dqn_orig":   ("models/dqn/best_model.zip",
                       "models/dqn/best_reward.txt"),
    }

    best_algo, best_reward = None, -np.inf
    print()
    for algo, (mpath, rpath) in candidates.items():
        if not os.path.exists(mpath): continue
        reward = -999.0
        if rpath and os.path.exists(rpath):
            try: reward = float(open(rpath).read().strip())
            except: pass
        print(f"  Found {algo:<18s}  reward={reward:>8.1f}")
        if reward > best_reward:
            best_reward, best_algo = reward, algo

    # Normalise aliases
    if best_algo == "ppo_orig":      best_algo = "ppo"
    if best_algo == "reinforce_orig": best_algo = "reinforce"
    if best_algo == "dqn_orig":      best_algo = "dqn"
    return best_algo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["random","best","dqn","ppo","ppo_v2","reinforce"],
                        default="random")
    parser.add_argument("--episodes",  type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"

    std_env = CropDroneEnv(
        grid_size=8, max_steps=200, max_fuel=100, max_payload=3,
        spread_interval=10, spread_prob=0.20, n_infected=5,
        render_mode=render_mode,
    )

    if args.mode == "random":
        run_random(std_env, episodes=args.episodes)

    elif args.mode == "ppo_v2":
        std_env.close()
        run_ppo_v2(render_mode)

    elif args.mode in ("dqn", "ppo"):
        run_sb3_model(std_env, args.mode)

    elif args.mode == "reinforce":
        run_reinforce_model(std_env)

    elif args.mode == "best":
        print("  Scanning all model folders...")
        best = find_best_model()
        if best is None:
            print("\n  No trained models found — running random agent.")
            run_random(std_env, episodes=args.episodes)
        else:
            print(f"\n  Best model: {best.upper()}")
            if best == "ppo_v2":
                std_env.close()
                run_ppo_v2(render_mode)
            elif best == "reinforce":
                run_reinforce_model(std_env)
            else:
                run_sb3_model(std_env, best)
