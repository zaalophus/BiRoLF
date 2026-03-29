"""
Jang_Experiment.py

Standalone experiment for:
  Jang et al. (ICML 2021)
  "Improved Regret Bounds of Bilinear Bandits using Action Space Analysis"

Setting
-------
  Pure bilinear bandit — no unobservable features, no subspace assumptions.
  User side : M=10 arms, each with d=5 observable features
  Item side : N=10 arms, each with d=5 observable features
  Expected reward : x_i^T Theta y_j

Agents
------
  epsilon-FALB  (JangEpsilonFALB)
  rO-UCB        (JangRoUCB)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import JangEpsilonFALB, JangRoUCB


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(agent, X, Y, Theta, horizon, noise_std):
    """
    X     : (M, d_x)
    Y     : (N, d_y)
    Theta : (d_x, d_y)
    Returns cumulative regret array of length horizon.
    """
    M, N = X.shape[0], Y.shape[0]
    exp_rewards = X @ Theta @ Y.T          # (M, N)
    max_abs = np.max(np.abs(exp_rewards))
    if max_abs > 1e-12:
        exp_rewards = exp_rewards / max_abs

    optimal = exp_rewards.max()
    regrets = np.zeros(horizon)

    for t in range(horizon):
        chosen  = agent.choose(X, Y)
        ci, cj  = chosen // N, chosen % N
        reward  = exp_rewards[ci, cj] + np.random.randn() * noise_std
        regrets[t] = optimal - exp_rewards[ci, cj]
        agent.update(X, Y, reward)

    return np.cumsum(regrets)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    M: int,
    N: int,
    d: int,
    horizon: int,
    trials: int,
    noise_std: float,
    seeds: list,
    jang_rank: int = None,
    delta: float   = 0.1,
):
    rank = jang_rank if jang_rank is not None else min(d, d)

    results = {"efalb": [], "roucb": []}

    for trial_idx in tqdm(range(trials), desc="Trials"):
        np.random.seed(seeds[trial_idx % len(seeds)] + 513 * trial_idx)

        X     = np.random.randn(M, d)           # (M, d)
        Y     = np.random.randn(N, d)           # (N, d)
        Theta = np.random.randn(d, d)           # (d, d)

        for key, agent in [
            ("efalb", JangEpsilonFALB(T=horizon, delta=delta, sigma=noise_std)),
            ("roucb", JangRoUCB(rank=rank, delta=delta, sigma=noise_std)),
        ]:
            cum_regret = run_trial(agent, X, Y, Theta, horizon, noise_std)
            results[key].append(cum_regret)

    return {k: np.array(v) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

AGENT_LABEL = {
    "efalb": r"$\epsilon$-FALB (Jang 2021)",
    "roucb": "rO-UCB (Jang 2021)",
}
COLORS = {
    "efalb": "#1f77b4",
    "roucb": "#ff7f0e",
}


def plot_results(results, horizon, save_path="Jang_Exp.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    rounds = np.arange(horizon)
    period = max(horizon // 10, 1)

    for i, (key, label) in enumerate(AGENT_LABEL.items()):
        data  = results[key]          # (trials, horizon)
        mean  = data.mean(axis=0)
        std   = data.std(axis=0, ddof=1) if data.shape[0] > 1 else np.zeros_like(mean)
        color = COLORS[key]

        marker_idx = rounds[i % period :: period]
        ax.errorbar(
            marker_idx, mean[marker_idx], yerr=std[marker_idx],
            fmt="s", color=color, linestyle="None",
            capsize=3, elinewidth=1,
            markeredgecolor="black", markeredgewidth=0.6,
        )
        ax.plot(rounds, mean, color=color, linewidth=2, label=label)

    ax.set_xlabel(r"Round ($t$)")
    ax.set_ylabel("Cumulative Regret")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    M, N      = 10, 10
    d         = 5
    horizon   = 10000
    trials    = 5
    noise_std = 0.1
    delta     = 0.1
    jang_rank = None        # None → min(d, d) = 5
    seeds     = [354]

    results = run_experiment(
        M=M, N=N, d=d,
        horizon=horizon,
        trials=trials,
        noise_std=noise_std,
        seeds=seeds,
        jang_rank=jang_rank,
        delta=delta,
    )

    plot_results(results, horizon, save_path="Jang_Exp.png")
