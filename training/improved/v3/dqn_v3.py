"""
training/improved/v3/dqn_v3.py
─────────────────────────────────────────────────────────────
DQN v3 — targets under -50 reward.

Root cause of all previous DQN failures:
  - Agent crashed (fuel=0) before finding any positive reward
  - Replay buffer filled with crash experiences → no useful Q-values
  - DQN trained on all-negative TD targets → Q-values never separated

Fix (mirrors PPO v2 exactly):
  1. Easy env   — fuel=200, payload=10, spread=OFF, n_infected=3
  2. RewardShapingWrapper — proximity signal guides navigation
  3. gradient_steps=8, train_freq=1 — maximum learning per env step
  4. learning_starts=3000 — let buffer fill with shaped experiences first
  5. 500k timesteps — enough for Q-values to meaningfully separate

Usage:
    python training/improved/v3/dqn_v3.py
    python training/improved/v3/dqn_v3.py --run 2
    python training/improved/v3/dqn_v3.py --smoke-test
"""

import os, sys, csv, time, argparse
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from environment.custom_env import CropDroneEnv
from training.improved.improved_ppo_v2 import RewardShapingWrapper, EASY_ENV_KWARGS

MODEL_DIR = "models/improved/v3/dqn"
LOG_DIR   = "logs/improved/v3/dqn"
CSV_PATH  = "models/improved/v3/dqn/dqn_v3_results.csv"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

TOTAL_TIMESTEPS = 500_000   # ~10–15 min per run on M1 Pro
EVAL_EPISODES   = 15
EVAL_FREQ       = 25_000


CONFIGS: List[Dict[str, Any]] = [
    # Run 1 — Baseline on easy env (the core fix)
    # gradient_steps=4 means 4 gradient updates per env step
    {
        "run_id":                   1,
        "description":              "Easy env baseline — lr=1e-4, grad=4, buf=100k, expl=0.25",
        "learning_rate":            1e-4,
        "buffer_size":              100_000,
        "learning_starts":          3_000,
        "batch_size":               64,
        "gamma":                    0.99,
        "train_freq":               1,
        "gradient_steps":           4,
        "exploration_fraction":     0.25,
        "exploration_final_eps":    0.02,
        "target_update_interval":   400,
        "net_arch":                 [256, 256],
        "proximity_scale":          2.0,
    },
    # Run 2 — Maximum gradient steps (8 per env step)
    # Most aggressive learning — extracts maximum from each experience
    {
        "run_id":                   2,
        "description":              "Max gradient — lr=1e-4, grad=8, buf=100k, expl=0.25",
        "learning_rate":            1e-4,
        "buffer_size":              100_000,
        "learning_starts":          3_000,
        "batch_size":               64,
        "gamma":                    0.99,
        "train_freq":               1,
        "gradient_steps":           8,
        "exploration_fraction":     0.25,
        "exploration_final_eps":    0.02,
        "target_update_interval":   400,
        "net_arch":                 [256, 256],
        "proximity_scale":          2.0,
    },
    # Run 3 — Stronger proximity reward (4.0)
    # Denser navigation signal → faster discovery of infected cells
    {
        "run_id":                   3,
        "description":              "Strong proximity — lr=1e-4, grad=4, prox=4.0, expl=0.25",
        "learning_rate":            1e-4,
        "buffer_size":              100_000,
        "learning_starts":          3_000,
        "batch_size":               64,
        "gamma":                    0.99,
        "train_freq":               1,
        "gradient_steps":           4,
        "exploration_fraction":     0.25,
        "exploration_final_eps":    0.02,
        "target_update_interval":   400,
        "net_arch":                 [256, 256],
        "proximity_scale":          4.0,
    },
    # Run 4 — Longer exploration on easy env
    # More epsilon-greedy steps before exploiting → discovers more cells
    {
        "run_id":                   4,
        "description":              "Long explore — lr=1e-4, grad=4, expl=0.40, prox=2.0",
        "learning_rate":            1e-4,
        "buffer_size":              100_000,
        "learning_starts":          3_000,
        "batch_size":               64,
        "gamma":                    0.99,
        "train_freq":               1,
        "gradient_steps":           4,
        "exploration_fraction":     0.40,
        "exploration_final_eps":    0.02,
        "target_update_interval":   400,
        "net_arch":                 [256, 256],
        "proximity_scale":          2.0,
    },
    # Run 5 — Best guess combo
    # Max gradients + strong proximity + moderate exploration
    {
        "run_id":                   5,
        "description":              "Best combo — lr=1e-4, grad=8, prox=3.0, expl=0.30, buf=150k",
        "learning_rate":            1e-4,
        "buffer_size":              150_000,
        "learning_starts":          5_000,
        "batch_size":               64,
        "gamma":                    0.99,
        "train_freq":               1,
        "gradient_steps":           8,
        "exploration_fraction":     0.30,
        "exploration_final_eps":    0.01,
        "target_update_interval":   300,
        "net_arch":                 [256, 256],
        "proximity_scale":          3.0,
    },
]


class TqdmCB(BaseCallback):
    def __init__(self, total: int, run_id: int):
        super().__init__()
        self.total = total; self.run_id = run_id; self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total, desc=f"  DQN v3 Run {self.run_id}/5",
            unit="step", ncols=92, colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    def _on_step(self):
        self.pbar.update(1)
        if self.num_timesteps % 25_000 == 0:
            self.pbar.set_postfix(eps=f"{self.model.exploration_rate:.3f}", refresh=False)
        return True

    def _on_training_end(self):
        if self.pbar: self.pbar.close()


def make_env_fn(seed: int = 0, proximity_scale: float = 2.0):
    def _init():
        base = CropDroneEnv(**EASY_ENV_KWARGS, render_mode=None, seed=seed)
        return Monitor(RewardShapingWrapper(base, proximity_scale=proximity_scale))
    return _init


def run_experiment(config: Dict, timesteps: int, seed: int) -> Dict:
    run_id = config["run_id"]
    prox   = config["proximity_scale"]
    print(f"\n{'='*62}")
    print(f"  DQN v3 — Run {run_id}/5")
    print(f"  {config['description']}")
    print(f"  easy_env=True | prox={prox} | grad_steps={config['gradient_steps']}")
    print(f"{'='*62}")

    run_dir = os.path.join(MODEL_DIR, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    train_env = make_vec_env(make_env_fn(seed=seed, proximity_scale=prox), n_envs=1)
    eval_env  = Monitor(
        RewardShapingWrapper(
            CropDroneEnv(**EASY_ENV_KWARGS, render_mode=None, seed=seed+100),
            proximity_scale=prox,
        )
    )

    model = DQN(
        policy                  = "MlpPolicy",
        env                     = train_env,
        learning_rate           = config["learning_rate"],
        buffer_size             = config["buffer_size"],
        learning_starts         = config["learning_starts"],
        batch_size              = config["batch_size"],
        gamma                   = config["gamma"],
        train_freq              = config["train_freq"],
        gradient_steps          = config["gradient_steps"],
        target_update_interval  = config["target_update_interval"],
        exploration_fraction    = config["exploration_fraction"],
        exploration_initial_eps = 1.0,
        exploration_final_eps   = config["exploration_final_eps"],
        policy_kwargs           = {"net_arch": config["net_arch"]},
        tensorboard_log         = os.path.join(LOG_DIR, f"run_{run_id:02d}"),
        verbose                 = 0,
        seed                    = seed,
        device                  = "cpu",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = run_dir,
        log_path             = run_dir,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = EVAL_EPISODES,
        deterministic        = True,
        verbose              = 0,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps,
                callback=[TqdmCB(timesteps, run_id), eval_cb],
                progress_bar=False)
    elapsed = time.time() - t0

    mean_r, std_r = evaluate_policy(model, eval_env,
                                    n_eval_episodes=EVAL_EPISODES, deterministic=True)
    model.save(os.path.join(run_dir, "final_model"))

    best_path = os.path.join(MODEL_DIR, "best_model")
    best_txt  = os.path.join(MODEL_DIR, "best_reward.txt")
    cur = -np.inf
    if os.path.exists(best_txt):
        try: cur = float(open(best_txt).read())
        except: pass
    if mean_r > cur:
        model.save(best_path)
        open(best_txt, "w").write(str(mean_r))
        print(f"  ★ New best DQN v3 (reward={mean_r:.2f})")

    train_env.close(); eval_env.close()
    print(f"\n  ✓ Run {run_id} — reward={mean_r:.2f}±{std_r:.2f}  time={elapsed/60:.1f}m")

    return {
        "run_id":               run_id,
        "description":          config["description"],
        "learning_rate":        config["learning_rate"],
        "buffer_size":          config["buffer_size"],
        "gradient_steps":       config["gradient_steps"],
        "exploration_fraction": config["exploration_fraction"],
        "proximity_scale":      prox,
        "easy_env":             True,
        "total_timesteps":      timesteps,
        "mean_reward":          round(mean_r, 2),
        "std_reward":           round(std_r, 2),
        "train_time_min":       round(elapsed / 60, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",        type=int, default=None)
    parser.add_argument("--timesteps",  type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        r = run_experiment(CONFIGS[0], timesteps=3_000, seed=42)
        print(f"  ✓ Smoke test passed  reward={r['mean_reward']}"); sys.exit(0)

    configs  = CONFIGS if args.run is None else [CONFIGS[args.run - 1]]
    results  = []

    print(f"\n  DQN v3  |  {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Timesteps: {args.timesteps:,}  |  Easy env + reward shaping")

    for cfg in configs:
        results.append(run_experiment(cfg, args.timesteps, args.seed))

    if len(results) > 1:
        print(f"\n{'='*62}  DQN v3 SUMMARY")
        best = max(r["mean_reward"] for r in results)
        for r in results:
            m = " ★" if r["mean_reward"] == best else "  "
            print(f"  Run {r['run_id']}{m}  reward={r['mean_reward']:>8.2f}  {r['description']}")
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader(); w.writerows(results)
        print(f"\n  Results → {CSV_PATH}")
        print(f"  TensorBoard → tensorboard --logdir {LOG_DIR}")
