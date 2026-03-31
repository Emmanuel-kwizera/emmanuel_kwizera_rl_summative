"""
training/improved/v3/pg_v3.py
─────────────────────────────────────────────────────────────
PPO v3 and REINFORCE v3 — pushing toward positive reward.

PPO v2 achieved -39. Target for v3: -10 to +50.
Strategy: build on what worked in v2 (easy env, proximity, VecNorm)
and push harder with more timesteps + slightly harder env to keep
learning meaningful past the easy baseline.

REINFORCE v2 achieved -69. Target for v3: -20 to -10.
Strategy: larger batch (32), more episodes (6000), run on easy env
same as v2 but with tuned hyperparameters around the best v2 run.

Usage:
    python training/improved/v3/pg_v3.py --algo ppo
    python training/improved/v3/pg_v3.py --algo reinforce
    python training/improved/v3/pg_v3.py --algo all
    python training/improved/v3/pg_v3.py --algo ppo --run 2
    python training/improved/v3/pg_v3.py --algo ppo --smoke-test
"""

import os, sys, csv, time, argparse
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from environment.custom_env import CropDroneEnv
from training.improved.improved_ppo_v2 import RewardShapingWrapper, EASY_ENV_KWARGS

PPO_DIR       = "models/improved/v3/ppo"
REINFORCE_DIR = "models/improved/v3/reinforce"
LOG_PPO       = "logs/improved/v3/ppo"
LOG_REIN      = "logs/improved/v3/reinforce"
for d in [PPO_DIR, REINFORCE_DIR, LOG_PPO, LOG_REIN]:
    os.makedirs(d, exist_ok=True)

PPO_TIMESTEPS      = 800_000   # ~14–20 min per run on M1 Pro
REINFORCE_EPISODES = 6_000     # ~5–8 min per run
EVAL_EPISODES      = 15
EVAL_FREQ          = 30_000
N_ENVS             = 4


# ═════════════════════════════════════════════════════════════
# PPO v3 CONFIGS — 5 runs pushing from -39 toward positive
# ═════════════════════════════════════════════════════════════
PPO_V3_CONFIGS: List[Dict[str, Any]] = [
    # Run 1 — Clone best v2 config (Run 2: prox=4.0) with 800k steps
    # v2 Run 2 was best at -39 with prox=4.0.
    # More timesteps alone should push past -39 toward -20 to 0.
    {
        "run_id":          1,
        "description":     "v2 best scaled — lr=1e-4, prox=4.0, n_steps=512, ent=0.01, epochs=5",
        "learning_rate":   1e-4,
        "n_steps":         512,
        "batch_size":      64,
        "n_epochs":        5,
        "gamma":           0.99,
        "gae_lambda":      0.95,
        "ent_coef":        0.01,
        "clip_range":      0.2,
        "vf_coef":         0.5,
        "net_arch":        [dict(pi=[256, 256], vf=[256, 256])],
        "proximity_scale": 4.0,
    },
    # Run 2 — Moderate proximity + more n_steps
    # Longer rollouts capture full treatment sequences (move→treat→move)
    {
        "run_id":          2,
        "description":     "Longer rollout — lr=1e-4, prox=3.0, n_steps=1024, ent=0.01, epochs=5",
        "learning_rate":   1e-4,
        "n_steps":         1024,
        "batch_size":      128,
        "n_epochs":        5,
        "gamma":           0.99,
        "gae_lambda":      0.95,
        "ent_coef":        0.01,
        "clip_range":      0.2,
        "vf_coef":         0.5,
        "net_arch":        [dict(pi=[256, 256], vf=[256, 256])],
        "proximity_scale": 3.0,
    },
    # Run 3 — Higher entropy to escape local optima
    # At -39, PPO may be stuck in a local optimum (patrol but don't treat).
    # More entropy forces exploration of treatment actions.
    {
        "run_id":          3,
        "description":     "High entropy — lr=1e-4, prox=4.0, n_steps=512, ent=0.05, epochs=5",
        "learning_rate":   1e-4,
        "n_steps":         512,
        "batch_size":      64,
        "n_epochs":        5,
        "gamma":           0.99,
        "gae_lambda":      0.95,
        "ent_coef":        0.05,
        "clip_range":      0.2,
        "vf_coef":         0.5,
        "net_arch":        [dict(pi=[256, 256], vf=[256, 256])],
        "proximity_scale": 4.0,
    },
    # Run 4 — Larger network + stronger proximity
    # More policy capacity + very strong proximity signal
    {
        "run_id":          4,
        "description":     "Large net — lr=1e-4, prox=5.0, n_steps=512, ent=0.01, arch=[256,256,128]",
        "learning_rate":   1e-4,
        "n_steps":         512,
        "batch_size":      64,
        "n_epochs":        5,
        "gamma":           0.99,
        "gae_lambda":      0.95,
        "ent_coef":        0.01,
        "clip_range":      0.2,
        "vf_coef":         0.5,
        "net_arch":        [dict(pi=[256, 256, 128], vf=[256, 256, 128])],
        "proximity_scale": 5.0,
    },
    # Run 5 — Best guess v3 combo
    # Moderate prox (not too strong — overpowered prox can mask treatment signal)
    # + moderate entropy + longer rollout + 8 epochs sweet spot
    {
        "run_id":          5,
        "description":     "Best combo — lr=1e-4, prox=3.5, n_steps=1024, ent=0.02, epochs=8",
        "learning_rate":   1e-4,
        "n_steps":         1024,
        "batch_size":      128,
        "n_epochs":        8,
        "gamma":           0.99,
        "gae_lambda":      0.95,
        "ent_coef":        0.02,
        "clip_range":      0.2,
        "vf_coef":         0.5,
        "net_arch":        [dict(pi=[256, 256, 128], vf=[256, 256, 128])],
        "proximity_scale": 3.5,
    },
]


# ═════════════════════════════════════════════════════════════
# REINFORCE v3 CONFIGS — 5 runs pushing from -69 toward -20
# ═════════════════════════════════════════════════════════════
REINFORCE_V3_CONFIGS: List[Dict[str, Any]] = [
    # Run 1 — Clone best v2 (Run 1: batch=16, lr=1e-3) with 6000 episodes
    {
        "run_id":         1,
        "description":    "v2 best scaled — lr=1e-3, hidden=128, batch=16, ent=0.01",
        "learning_rate":  1e-3,
        "gamma":          0.99,
        "hidden_dim":     128,
        "entropy_coef":   0.01,
        "batch_episodes": 16,
        "reward_clip":    50.0,
        "normalize":      True,
        "grad_clip":      1.0,
        "proximity_scale":2.0,
    },
    # Run 2 — Maximum batch (32 episodes)
    # Most stable gradient estimate possible for vanilla PG
    {
        "run_id":         2,
        "description":    "Max batch=32 — lr=1e-3, hidden=128, ent=0.01",
        "learning_rate":  1e-3,
        "gamma":          0.99,
        "hidden_dim":     128,
        "entropy_coef":   0.01,
        "batch_episodes": 32,
        "reward_clip":    50.0,
        "normalize":      True,
        "grad_clip":      1.0,
        "proximity_scale":2.0,
    },
    # Run 3 — Strong proximity + large batch
    # Denser navigation signal + stable gradient
    {
        "run_id":         3,
        "description":    "Strong prox=4.0 — lr=1e-3, hidden=128, batch=16, ent=0.01",
        "learning_rate":  1e-3,
        "gamma":          0.99,
        "hidden_dim":     128,
        "entropy_coef":   0.01,
        "batch_episodes": 16,
        "reward_clip":    50.0,
        "normalize":      True,
        "grad_clip":      1.0,
        "proximity_scale":4.0,
    },
    # Run 4 — Higher entropy + large batch
    # More exploration of treatment actions + stable gradient
    {
        "run_id":         4,
        "description":    "High entropy — lr=1e-3, hidden=128, batch=16, ent=0.05, prox=3.0",
        "learning_rate":  1e-3,
        "gamma":          0.99,
        "hidden_dim":     128,
        "entropy_coef":   0.05,
        "batch_episodes": 16,
        "reward_clip":    50.0,
        "normalize":      True,
        "grad_clip":      1.0,
        "proximity_scale":3.0,
    },
    # Run 5 — Best guess v3 combo
    {
        "run_id":         5,
        "description":    "Best combo — lr=8e-4, hidden=128, batch=32, ent=0.02, prox=3.0",
        "learning_rate":  8e-4,
        "gamma":          0.99,
        "hidden_dim":     128,
        "entropy_coef":   0.02,
        "batch_episodes": 32,
        "reward_clip":    50.0,
        "normalize":      True,
        "grad_clip":      1.0,
        "proximity_scale":3.0,
    },
]


# ═════════════════════════════════════════════════════════════
# PPO TRAINING
# ═════════════════════════════════════════════════════════════
class TqdmPPO(BaseCallback):
    def __init__(self, total: int, run_id: int):
        super().__init__()
        self.total = total; self.run_id = run_id; self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total, desc=f"  PPO v3 Run {self.run_id}/5",
                         unit="step", ncols=92, colour="cyan",
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    def _on_step(self):
        self.pbar.update(1); return True

    def _on_training_end(self):
        if self.pbar: self.pbar.close()


def make_ppo_env(seed: int = 0, proximity_scale: float = 2.0):
    def _init():
        base = CropDroneEnv(**EASY_ENV_KWARGS, render_mode=None, seed=seed)
        return Monitor(RewardShapingWrapper(base, proximity_scale=proximity_scale))
    return _init


def run_ppo_v3(config: Dict, timesteps: int, seed: int) -> Dict:
    run_id = config["run_id"]
    prox   = config["proximity_scale"]
    print(f"\n{'='*62}")
    print(f"  PPO v3 — Run {run_id}/5")
    print(f"  {config['description']}")
    print(f"  prox={prox} | VecNormalize=True | n_envs={N_ENVS}")
    print(f"{'='*62}")

    run_dir = os.path.join(PPO_DIR, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    train_vec = make_vec_env(make_ppo_env(seed=seed, proximity_scale=prox), n_envs=N_ENVS)
    train_env = VecNormalize(train_vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_vec  = make_vec_env(make_ppo_env(seed=seed+100, proximity_scale=prox), n_envs=1)
    eval_env  = VecNormalize(eval_vec, norm_obs=True, norm_reward=False,
                             clip_obs=10.0, training=False)

    model = PPO(
        policy              = "MlpPolicy",
        env                 = train_env,
        learning_rate       = config["learning_rate"],
        n_steps             = config["n_steps"],
        batch_size          = config["batch_size"],
        n_epochs            = config["n_epochs"],
        gamma               = config["gamma"],
        gae_lambda          = config["gae_lambda"],
        ent_coef            = config["ent_coef"],
        clip_range          = config["clip_range"],
        vf_coef             = config["vf_coef"],
        normalize_advantage = True,
        policy_kwargs       = {"net_arch": config["net_arch"]},
        tensorboard_log     = os.path.join(LOG_PPO, f"run_{run_id:02d}"),
        verbose             = 0,
        seed                = seed,
        device              = "cpu",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = run_dir,
        log_path             = run_dir,
        eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes      = EVAL_EPISODES,
        deterministic        = True,
        verbose              = 0,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps,
                callback=[TqdmPPO(timesteps, run_id), eval_cb],
                progress_bar=False)
    elapsed = time.time() - t0

    eval_env.obs_rms = train_env.obs_rms
    mean_r, std_r = evaluate_policy(model, eval_env,
                                    n_eval_episodes=EVAL_EPISODES, deterministic=True)

    model.save(os.path.join(run_dir, "final_model"))
    train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))

    best_path = os.path.join(PPO_DIR, "best_model")
    best_txt  = os.path.join(PPO_DIR, "best_reward.txt")
    cur = -np.inf
    if os.path.exists(best_txt):
        try: cur = float(open(best_txt).read())
        except: pass
    if mean_r > cur:
        model.save(best_path)
        train_env.save(os.path.join(PPO_DIR, "vecnormalize_best.pkl"))
        open(best_txt, "w").write(str(mean_r))
        print(f"  ★ New best PPO v3 (reward={mean_r:.2f})")

    train_env.close(); eval_env.close()
    print(f"\n  ✓ PPO v3 Run {run_id} — reward={mean_r:.2f}±{std_r:.2f}  time={elapsed/60:.1f}m")

    return {
        "run_id":           run_id,
        "description":      config["description"],
        "learning_rate":    config["learning_rate"],
        "n_steps":          config["n_steps"],
        "n_envs":           N_ENVS,
        "n_epochs":         config["n_epochs"],
        "ent_coef":         config["ent_coef"],
        "proximity_scale":  prox,
        "vec_normalize":    True,
        "total_timesteps":  timesteps,
        "mean_reward":      round(mean_r, 2),
        "std_reward":       round(std_r, 2),
        "train_time_min":   round(elapsed / 60, 2),
    }


# ═════════════════════════════════════════════════════════════
# REINFORCE TRAINING
# ═════════════════════════════════════════════════════════════
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

    def act(self, obs: np.ndarray):
        t = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.forward(t)
        dist  = Categorical(probs)
        a     = dist.sample()
        return a.item(), dist.log_prob(a), dist.entropy()


def compute_returns(rewards: list, gamma: float) -> torch.Tensor:
    G, ret = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        ret.insert(0, G)
    return torch.FloatTensor(ret)


def make_rein_env(seed: int, proximity_scale: float) -> CropDroneEnv:
    base = CropDroneEnv(**EASY_ENV_KWARGS, render_mode=None, seed=seed)
    return RewardShapingWrapper(base, proximity_scale=proximity_scale)


def run_reinforce_v3(config: Dict, total_episodes: int, seed: int) -> Dict:
    run_id = config["run_id"]
    prox   = config["proximity_scale"]
    print(f"\n{'='*62}")
    print(f"  REINFORCE v3 — Run {run_id}/5")
    print(f"  {config['description']}")
    print(f"  prox={prox} | batch={config['batch_episodes']} | clip={config['reward_clip']}")
    print(f"{'='*62}")

    run_dir = os.path.join(REINFORCE_DIR, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(seed); np.random.seed(seed)
    env = make_rein_env(seed=seed, proximity_scale=prox)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n

    policy    = PolicyNet(obs_dim, n_actions, config["hidden_dim"])
    optimizer = optim.Adam(policy.parameters(), lr=config["learning_rate"])

    batch   = config["batch_episodes"]
    ep_rews = []
    best    = -np.inf
    t0      = time.time()

    pbar = tqdm(
        range(0, total_episodes, batch),
        desc=f"  REINFORCE v3 Run {run_id}/5",
        unit=f"batch({batch})", ncols=92, colour="magenta",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    for _ in pbar:
        all_lp, all_ret, all_ent = [], [], []

        for _ in range(batch):
            obs, _ = env.reset()
            lps, rwds, ents = [], [], []
            done = False
            while not done:
                a, lp, ent = policy.act(obs)
                obs, r, te, tr, _ = env.step(a)
                rwds.append(float(np.clip(r, -config["reward_clip"], config["reward_clip"])))
                lps.append(lp); ents.append(ent)
                done = te or tr
            ep_rews.append(sum(rwds))

            returns = compute_returns(rwds, config["gamma"])
            if config["normalize"] and returns.std() > 1e-6:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            all_lp.extend(lps); all_ret.extend(returns.tolist()); all_ent.extend(ents)

        lp_t  = torch.stack(all_lp)
        ret_t = torch.FloatTensor(all_ret)
        ent_t = torch.stack(all_ent)
        loss  = -(lp_t * ret_t).mean() - config["entropy_coef"] * ent_t.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config["grad_clip"])
        optimizer.step()

        window = min(len(ep_rews), batch * 5)
        recent = float(np.mean(ep_rews[-window:]))
        pbar.set_postfix(reward=f"{recent:.1f}", loss=f"{loss.item():.3f}", refresh=False)

        if recent > best:
            best = recent
            torch.save(policy.state_dict(), os.path.join(run_dir, "best_policy.pt"))

    pbar.close()
    elapsed = time.time() - t0

    # Greedy eval
    eval_env = make_rein_env(seed=seed+300, proximity_scale=prox)
    eval_rs  = []
    policy.eval()
    with torch.no_grad():
        for _ in range(EVAL_EPISODES):
            obs, _ = eval_env.reset()
            ep_r, done = 0.0, False
            while not done:
                probs  = policy(torch.FloatTensor(obs).unsqueeze(0))
                action = probs.argmax(dim=-1).item()
                obs, r, te, tr, _ = eval_env.step(action)
                ep_r += r; done = te or tr
            eval_rs.append(ep_r)
    eval_env.close(); env.close()

    mean_r = float(np.mean(eval_rs))
    std_r  = float(np.std(eval_rs))
    torch.save(policy.state_dict(), os.path.join(run_dir, "final_policy.pt"))

    best_path = os.path.join(REINFORCE_DIR, "reinforce_v3_best.pt")
    best_txt  = os.path.join(REINFORCE_DIR, "reinforce_v3_best_reward.txt")
    cur = -np.inf
    if os.path.exists(best_txt):
        try: cur = float(open(best_txt).read())
        except: pass
    if mean_r > cur:
        torch.save(policy.state_dict(), best_path)
        open(best_txt, "w").write(str(mean_r))
        print(f"  ★ New best REINFORCE v3 (reward={mean_r:.2f})")

    print(f"\n  ✓ REINFORCE v3 Run {run_id} — reward={mean_r:.2f}±{std_r:.2f}  time={elapsed/60:.1f}m")

    return {
        "run_id":          run_id,
        "description":     config["description"],
        "learning_rate":   config["learning_rate"],
        "hidden_dim":      config["hidden_dim"],
        "entropy_coef":    config["entropy_coef"],
        "batch_episodes":  config["batch_episodes"],
        "proximity_scale": prox,
        "easy_env":        True,
        "total_episodes":  total_episodes,
        "mean_reward":     round(mean_r, 2),
        "std_reward":      round(std_r, 2),
        "train_time_min":  round(elapsed / 60, 2),
    }


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",       choices=["ppo","reinforce","all"], default="all")
    parser.add_argument("--run",        type=int,  default=None)
    parser.add_argument("--ppo-steps",  type=int,  default=PPO_TIMESTEPS)
    parser.add_argument("--rein-eps",   type=int,  default=REINFORCE_EPISODES)
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    def pick(cfgs):
        if args.run is not None:
            idx = args.run - 1
            assert 0 <= idx < len(cfgs)
            return [cfgs[idx]]
        return cfgs

    smoke = args.smoke_test
    print(f"\n  PG v3  |  {datetime.now().strftime('%H:%M:%S')}  |  algo={args.algo}")

    if args.algo in ("ppo", "all"):
        steps = 3_000 if smoke else args.ppo_steps
        ppo_res = [run_ppo_v3(c, steps, args.seed) for c in pick(PPO_V3_CONFIGS)]
        if len(ppo_res) > 1:
            print(f"\n{'='*62}  PPO v3 SUMMARY")
            best = max(r["mean_reward"] for r in ppo_res)
            for r in ppo_res:
                m = " ★" if r["mean_reward"] == best else "  "
                print(f"  Run {r['run_id']}{m}  reward={r['mean_reward']:>8.2f}  {r['description']}")
            path = os.path.join(PPO_DIR, "ppo_v3_results.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=ppo_res[0].keys())
                w.writeheader(); w.writerows(ppo_res)
            print(f"  Results → {path}")

    if args.algo in ("reinforce", "all"):
        eps = 32 if smoke else args.rein_eps
        rein_res = [run_reinforce_v3(c, eps, args.seed) for c in pick(REINFORCE_V3_CONFIGS)]
        if len(rein_res) > 1:
            print(f"\n{'='*62}  REINFORCE v3 SUMMARY")
            best = max(r["mean_reward"] for r in rein_res)
            for r in rein_res:
                m = " ★" if r["mean_reward"] == best else "  "
                print(f"  Run {r['run_id']}{m}  reward={r['mean_reward']:>8.2f}  {r['description']}")
            path = os.path.join(REINFORCE_DIR, "reinforce_v3_results.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rein_res[0].keys())
                w.writeheader(); w.writerows(rein_res)
            print(f"  Results → {path}")
