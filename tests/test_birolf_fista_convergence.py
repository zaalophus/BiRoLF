import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import BiRoLFLasso


def run_convergence_check(seed: int = 0, M: int = 6, N: int = 5, rounds: int = 8):
    rng = np.random.default_rng(seed)
    # Simple, well-conditioned augmented features
    x = np.eye(M, dtype=float)
    y = np.eye(N, dtype=float)
    # Ground-truth Phi for reward generation
    phi_star = rng.standard_normal((M, N)) * 0.5

    agent = BiRoLFLasso(
        M=M,
        N=N,
        sigma=0.1,
        delta=0.1,
        p=1.0,  # force match probability to 1
        lam_c_impute=0.01,
        lam_c_main=0.01,
        fista_max_iter=1000,
        fista_tol=1e-8,
    )

    kkt_vals = []
    for t in range(1, rounds + 1):
        agent.t = t
        # choose a random action, force match
        i = rng.integers(0, M)
        j = rng.integers(0, N)
        agent.chosen_action = i * N + j
        agent.pseudo_action = agent.chosen_action
        r = float(phi_star[i, j])
        agent.update(x=x, y=y, r=r)
        kkt_vals.append(agent.main_kkt_violation())

    return kkt_vals


def test_birolf_fista_convergence():
    kkt_vals = run_convergence_check(seed=0)
    # Expect small KKT residuals with high-iter FISTA
    assert max(kkt_vals) < 1e-4, f"KKT too large: {kkt_vals}"


if __name__ == "__main__":
    vals = run_convergence_check(seed=0)
    print("KKT residuals:", [f"{v:.3e}" for v in vals])
    if max(vals) < 1e-4:
        print("PASS: BiRoLFLasso FISTA appears to converge (KKT < 1e-4).")
    else:
        print("FAIL: KKT residuals are larger than expected.")
