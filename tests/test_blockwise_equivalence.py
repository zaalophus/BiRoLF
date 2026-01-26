import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import (
    build_augmented_gram,
    main_objective,
    solve_main_blockwise,
    fista_lasso_matrix,
    BiRoLFLasso,
)
from util import pseudo_action_match_rate


def _make_spd(n: int, rng: np.random.Generator) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    A = rng.standard_normal((n, n))
    return A.T @ A + 0.1 * np.eye(n)


def test_pseudo_action_distribution():
    total_arms = 50
    chosen = 7
    p = 0.6
    rng = np.random.default_rng(0)
    rate = pseudo_action_match_rate(total_arms, p, chosen, trials=20000, rng=rng)
    assert abs(rate - p) < 0.02


def test_blockwise_vs_full_objective():
    rng = np.random.default_rng(1)
    M, N = 8, 7
    dx, dy = 3, 2
    Gx_o = _make_spd(dx, rng)
    Gy_o = _make_spd(dy, rng)
    B = rng.standard_normal((M, N))
    mu = 0.05

    Gx = build_augmented_gram(Gx_o, M, dx)
    Gy = build_augmented_gram(Gy_o, N, dy)
    Phi0 = np.zeros((M, N), dtype=float)

    Phi_full = fista_lasso_matrix(
        Gx,
        Gy,
        B,
        mu,
        Phi0,
        L=2.0 * max(np.linalg.eigvalsh(Gx).max(), 1e-12) * max(np.linalg.eigvalsh(Gy).max(), 1e-12),
        max_iter=500,
        tol=1e-8,
        use_fista=True,
    )
    Phi_block = solve_main_blockwise(
        B=B,
        Gx_o=Gx_o,
        Gy_o=Gy_o,
        mu=mu,
        dx=dx,
        dy=dy,
        Phi_init=Phi0,
        params={
            "block_oo_max_iter": 500,
            "block_ou_max_iter": 300,
            "block_uo_max_iter": 300,
            "block_tol": 1e-8,
            "block_use_fista": True,
            "block_use_batched": True,
        },
        lam_x_max=float(np.linalg.eigvalsh(Gx_o).max()),
        lam_y_max=float(np.linalg.eigvalsh(Gy_o).max()),
    )

    obj_full = main_objective(Phi_full, Gx, Gy, B, mu)
    obj_block = main_objective(Phi_block, Gx, Gy, B, mu)
    denom = max(1.0, abs(obj_full))
    assert abs(obj_full - obj_block) / denom <= 1e-4

    denom_phi = max(1.0, np.linalg.norm(Phi_full))
    assert np.linalg.norm(Phi_full - Phi_block) / denom_phi <= 1e-3


def test_impute_grad_vectorized():
    rng = np.random.default_rng(2)
    M, N = 6, 5
    dx, dy = 4, 3
    X = rng.standard_normal((M, dx))
    Y = rng.standard_normal((N, dy))
    Phi = rng.standard_normal((dx, dy))
    Ncnt = rng.integers(low=0, high=5, size=(M, N))
    Ssum = rng.standard_normal((M, N))
    C_sum = X.T @ Ssum @ Y

    agent = BiRoLFLasso(
        M=M,
        N=N,
        sigma=0.1,
        delta=0.1,
        p=0.6,
        lam_c_impute=1.0,
        lam_c_main=1.0,
    )
    agent._init_static_arms_if_needed(X, Y)
    agent.Ncnt = Ncnt
    agent.Ssum = Ssum
    agent.C_sum = C_sum

    grad_new = agent._grad_impute(Phi)
    grad_ref = np.zeros_like(Phi)
    for i in range(M):
        xi = X[i, :]
        for j in range(N):
            yj = Y[j, :]
            sij = float(xi @ Phi @ yj)
            coeff = 2.0 * (Ncnt[i, j] * sij - Ssum[i, j])
            grad_ref += coeff * np.outer(xi, yj)

    denom = max(1.0, np.linalg.norm(grad_ref))
    assert np.linalg.norm(grad_new - grad_ref) / denom <= 1e-8


if __name__ == "__main__":
    test_pseudo_action_distribution()
    test_blockwise_vs_full_objective()
    test_impute_grad_vectorized()
    print("All tests passed.")
