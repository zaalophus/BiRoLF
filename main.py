from cfg import get_cfg
from util import *
import time
import datetime
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pickle
import os

MOTHER_PATH = "."

DIST_DICT = {"gaussian": "g", "uniform": "u"}

AGENT_DICT = {
    "mab_ucb": r"UCB($\delta$)",
    "linucb": "LinUCB",
    "lints": "LinTS",
    "rolf_lasso": "RoLF-Lasso (Kim & Park)",
    "rolf_ridge": "RoLF-Ridge (Kim & Park)",
    "birolf_lasso": "BiRoLF-Lasso (Ours)",
    "birolf_lasso_fista": "BiRoLF-Lasso-FISTA (Ours)",
    "birolf_lasso_blockwise": "BiRoLF-Lasso-Blockwise (Ours)",
    "estr_lowoful": "ESTR+LowOFUL",
    "dr_lasso": "DRLasso",
}

cfg = get_cfg()

def _maybe_set_blas_threads():
    if not getattr(cfg, "set_blas_threads", False):
        return
    n_threads = int(getattr(cfg, "blas_threads_per_worker", 1))
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n_threads)
    except Exception:
        pass
# --- Regularization multipliers (can be provided via cfg if added) ---
lam_c_impute = getattr(cfg, "lam_c_impute", None)
lam_c_main = getattr(cfg, "lam_c_main", None)
# Fall back to model defaults if None
if lam_c_impute is None:
    lam_c_impute = 1.0
if lam_c_main is None:
    lam_c_main = 0.1  # bilinear-friendly default; RoLFLasso can override via CLI
# Per-algorithm pass-through (if cfg is None, keep algorithm-specific defaults)
lam_c_impute_lin = lam_c_impute if lam_c_impute is not None else 1.0
lam_c_main_lin   = lam_c_main   if lam_c_main   is not None else 1.0
lam_c_impute_bilin = lam_c_impute if lam_c_impute is not None else 1.0
lam_c_main_bilin   = lam_c_main   if lam_c_main   is not None else 0.1

date = datetime.datetime.now().strftime('%Y-%m-%d')
RUN_TAG = dt.now().strftime("%H%M%S_%f")

RESULT_PATH = f"{MOTHER_PATH}/results/{date}/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"
FIGURE_PATH = f"{MOTHER_PATH}/figures/{date}/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"
LOG_PATH = (
    f"{MOTHER_PATH}/logs/{date}/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"
)

# Global timing tracking variables
TIMING_DATA = {}  # {agent_name: {trial: [update_times]}}
TOTAL_EXECUTION_TIMES = {}  # {agent_name: [total_times_per_trial]}

# Import models after setting up TIMING_DATA
import models
from models import *

# Share timing data with models (if models has TIMING_DATA attribute)
if hasattr(models, 'TIMING_DATA'):
    models.TIMING_DATA = TIMING_DATA
import models
models.TIMING_DATA = TIMING_DATA

## ~! Generate feature matrix Z(full feature), X(observerable feature) !~


# Case 1: default, default
# Case 2: default, R(V) ⊆ R(Y)
# Case 3: default, R(Y) ⊆ R(V)

# Case 4: R(U) ⊆ R(X), default
# Case 5: R(U) ⊆ R(X), R(V) ⊆ R(Y)
# Case 6: R(U) ⊆ R(X), R(Y) ⊆ R(V)

# Case 7: R(X) ⊆ R(U), default
# Case 8: R(X) ⊆ R(U), R(V) ⊆ R(Y)
# Case 9: R(X) ⊆ R(U), R(Y) ⊆ R(V)


def bilinear_feature_generator(
    case: int,
    d_x_star: int,
    d_x: int,
    d_y_star: int,
    d_y: int,
    M: int,
    N: int,
):
    ## sample the true, observable, and unobservable features
    d_u = d_x_star - d_x  # dimension of unobserved feature(x)
    d_v = d_y_star - d_y  # dimension of unobserved feature(y)

    assert case in [1, 2, 3, 4, 5, 6, 7, 8, 9], "There exists only Case 1 to 9."

    ## For feature x
    if case in [1, 2, 3]:
        ## X is Default case
        X_star = np.random.multivariate_normal(
            mean=np.zeros(d_x_star), cov=np.eye(d_x_star), size=M
        ).T  # (d_x_star, M)
        X = X_star[:d_x, :]  # (d_x, M)

    # For two matrices A and B,
    # if each row of A can be expressed as a linear combination of the rows of B,
    # then R(A) ⊆ R(B)
    elif case in [4, 5, 6]:

        ## ~! When observable feature dominates !~

        ## R(U) ⊆ R(X)
        # First generate X
        X = np.random.multivariate_normal(
            mean=np.zeros(d_x), cov=np.eye(d_x), size=M
        ).T  # (d_x, M)

        # Generate a coefficient matrix C
        # C = np.random.multivariate_normal(mean=np.zeros(d_x),
        #                                   cov=np.eye(d_x),
        #                                   size=d_u).T # (d_u, d_x)
        C = np.random.uniform(
            low=-1 / np.pi, high=1 / np.pi, size=(d_u, d_x)
        )  # (d_u, d_x)

        # Compute U as a multiplication between C and X
        U = C @ X  # (d_u, M)
        X_star = np.concatenate([X, U], axis=0)  # (d_x_star, M)))

    elif case in [7, 8, 9]:

        ## ~! When the unobservable feature dominates !~

        ## R(X) ⊆ R(U)
        # First generate U
        U = np.random.multivariate_normal(
            mean=np.zeros(d_u), cov=np.eye(d_u), size=M
        ).T  # (d_u, M)

        # Generate a coefficient matrix C
        # C = np.random.multivariate_normal(mean=np.zeros(d_x),
        #                                   cov=np.eye(d_x),
        #                                   size=d_u).T # (d_x, d_u)
        C = np.random.uniform(
            low=-1 / np.pi, high=1 / np.pi, size=(d_x, d_u)
        )  # (d_x, d_u)

        # Compute U as a multiplication between C and X
        X = C @ U  # (d_x, M)
        X_star = np.concatenate([X, U], axis=0)  # (d_x_star, M)))

    ## For feature y,
    if case in [1, 4, 7]:
        ## Y is Default case
        Y_star = np.random.multivariate_normal(
            mean=np.zeros(d_y_star), cov=np.eye(d_y_star), size=N
        ).T  # (d_y_star, N)
        Y = Y_star[:d_y, :]  # (d_y, N)

    # For two matrices A and B,
    # if each row of A can be expressed as a linear combination of the rows of B,
    # then R(A) ⊆ R(B)
    elif case in [2, 5, 8]:

        ## ~! When observable feature dominates !~

        ## R(V) ⊆ R(Y)

        # First generate Y
        Y = np.random.multivariate_normal(
            mean=np.zeros(d_y), cov=np.eye(d_y), size=N
        ).T  # (d_y, N)

        # Generate a coefficient matrix C
        # C = np.random.multivariate_normal(mean=np.zeros(d_y),
        #                                   cov=np.eye(d_y),
        #                                   size=d_v).T # (d_v, d_y)
        C = np.random.uniform(
            low=-1 / np.pi, high=1 / np.pi, size=(d_v, d_y)
        )  # (d_v, d_y)

        # Compute U as a multiplication between C and Y
        V = C @ Y  # (d_v, N)
        Y_star = np.concatenate([Y, V], axis=0)  # (d_y_star, N)))

    elif case in [3, 6, 9]:

        ## ~! When the unobservable feature dominates !~

        ## R(Y) ⊆ R(V)
        # First generate V
        V = np.random.multivariate_normal(
            mean=np.zeros(d_v), cov=np.eye(d_v), size=N
        ).T  # (d_v, N)

        # Generate a coefficient matrix C
        # C = np.random.multivariate_normal(mean=np.zeros(d_y),
        #                                   cov=np.eye(d_y),
        #                                   size=d_v).T # (d_y, d_v)
        C = np.random.uniform(
            low=-1 / np.pi, high=1 / np.pi, size=(d_y, d_v)
        )  # (d_y, d_v)

        # Compute U as a multiplication between C and X
        Y = C @ V  # (d_y, N)
        Y_star = np.concatenate([Y, V], axis=0)  # (d_y_star, N)))

    return X_star, X, Y_star, Y


def bilinear_run_trial(
    agent_type: str,
    now_trial: int,
    horizon: int,
    d_x_star: int,
    d_x: int,
    M: int,
    d_y_star: int,
    d_y: int,
    N: int,
    noise_std: float,
    case: int,
    verbose: bool,
    fname: str,
    timing_data: dict = None,
):
    total_arms = M * N
    total_obs_dim = d_x * d_y

    ## how much do the exploration phase?
    exp_map = {
        "double": (2 * (M * N)),
        "sqr": ((M * N) ** 2),
        "K": (M * N),
        "triple": (3 * (M * N)),
        "quad": (4 * (M * N)),
    }

    ## run and collect the regrets
    regret_container = np.zeros(1, dtype=object)
    
    ### Setting random state (Manual Folded)

    ### Select agent (Manual Folded)
    if agent_type == "linucb":
        agent = LinUCB(d=total_obs_dim, lbda=cfg.p, delta=cfg.delta)

    elif agent_type == "lints":
        agent = LinTS(
            d=total_obs_dim,
            lbda=cfg.p,
            horizon=horizon,
            reward_std=noise_std,
            delta=cfg.delta,
        )

    elif agent_type == "mab_ucb":
        agent = UCBDelta(n_arms=total_arms, delta=cfg.delta)

    elif agent_type == "rolf_lasso":
        if cfg.explore:
            agent = RoLFLasso(
                d=total_obs_dim,
                arms=total_arms,
                p=cfg.p,
                delta=cfg.delta,
                sigma=noise_std,
                explore=cfg.explore,
                init_explore=exp_map[cfg.init_explore],
                lam_c_impute=cfg.lamc_rolf_impute,
                lam_c_main=cfg.lamc_rolf_main,
            )
        else:
            agent = RoLFLasso(
                d=total_obs_dim,
                arms=total_arms,
                p=cfg.p,
                delta=cfg.delta,
                sigma=noise_std,
                lam_c_impute=cfg.lamc_rolf_impute,
                lam_c_main=cfg.lamc_rolf_main,
            )

    elif agent_type == "rolf_ridge":
        if cfg.explore:
            agent = RoLFRidge(
                d=total_obs_dim,
                arms=total_arms,
                p=cfg.p,
                delta=cfg.delta,
                sigma=noise_std,
                explore=cfg.explore,
                init_explore=exp_map[cfg.init_explore],
            )
        else:
            agent = RoLFRidge(
                d=total_obs_dim,
                arms=total_arms,
                p=cfg.p,
                delta=cfg.delta,
                sigma=noise_std,
            )

    elif agent_type == "dr_lasso":
        agent = DRLassoBandit(
            d=total_obs_dim, arms=total_arms, lam1=1.0, lam2=0.5, zT=10, tr=True
        )

    elif agent_type == "birolf_lasso":
        if cfg.explore:
            agent = BiRoLFLasso(
                M=M,
                N=N,
                sigma=noise_std,
                delta=cfg.delta,
                p=cfg.p,
                p1=cfg.p1,
                p2=cfg.p2,
                explore=cfg.explore,
                init_explore=exp_map[cfg.init_explore],
                theoretical_init_explore=False,
                lam_c_impute=cfg.lamc_bi_impute,
                lam_c_main=cfg.lamc_bi_main,
            )
        else:
            agent = BiRoLFLasso(
                M=M,
                N=N,
                sigma=noise_std,
                delta=cfg.delta,
                p=cfg.p,
                p1=cfg.p1,
                p2=cfg.p2,
                theoretical_init_explore=False,
                lam_c_impute=cfg.lamc_bi_impute,
                lam_c_main=cfg.lamc_bi_main,
            )

    elif agent_type == "birolf_lasso_fista":
        if cfg.explore:
            agent = BiRoLFLasso_FISTA(
                M=M,
                N=N,
                sigma=noise_std,
                delta=cfg.delta,
                p=cfg.p,
                p1=cfg.p1,
                p2=cfg.p2,
                explore=cfg.explore,
                init_explore=exp_map[cfg.init_explore],
                theoretical_init_explore=False,
                lam_c_impute=cfg.lamc_bi_impute,
                lam_c_main=cfg.lamc_bi_main,
            )
        else:
            agent = BiRoLFLasso_FISTA(
                M=M,
                N=N,
                sigma=noise_std,
                delta=cfg.delta,
                p=cfg.p,
                p1=cfg.p1,
                p2=cfg.p2,
                theoretical_init_explore=False,
                lam_c_impute=cfg.lamc_bi_impute,
                lam_c_main=cfg.lamc_bi_main,
            )

    elif agent_type == "birolf_lasso_blockwise":
        if cfg.explore:
            agent = BiRoLFLasso_Blockwise(
                M=M,
                N=N,
                d_x=d_x,
                d_y=d_y,
                sigma=noise_std,
                delta=cfg.delta,
                p=cfg.p,
                p1=cfg.p1,
                p2=cfg.p2,
                explore=cfg.explore,
                init_explore=exp_map[cfg.init_explore],
                theoretical_init_explore=False,
                lam_c_impute=cfg.lamc_bi_impute,
                lam_c_main=cfg.lamc_bi_main,
                block_oo_max_iter=getattr(cfg, "block_oo_max_iter", 100),
                block_ou_max_iter=getattr(cfg, "block_ou_max_iter", 50),
                block_uo_max_iter=getattr(cfg, "block_uo_max_iter", 50),
                block_tol=getattr(cfg, "block_tol", 1e-6),
                block_use_fista=getattr(cfg, "block_use_fista", True),
                block_use_batched=getattr(cfg, "block_use_batched", True),
            )
        else:
            agent = BiRoLFLasso_Blockwise(
                M=M,
                N=N,
                d_x=d_x,
                d_y=d_y,
                sigma=noise_std,
                delta=cfg.delta,
                p=cfg.p,
                p1=cfg.p1,
                p2=cfg.p2,
                theoretical_init_explore=False,
                lam_c_impute=cfg.lamc_bi_impute,
                lam_c_main=cfg.lamc_bi_main,
                block_oo_max_iter=getattr(cfg, "block_oo_max_iter", 100),
                block_ou_max_iter=getattr(cfg, "block_ou_max_iter", 50),
                block_uo_max_iter=getattr(cfg, "block_uo_max_iter", 50),
                block_tol=getattr(cfg, "block_tol", 1e-6),
                block_use_fista=getattr(cfg, "block_use_fista", True),
                block_use_batched=getattr(cfg, "block_use_batched", True),
            )

    elif agent_type == "estr_lowoful":
        agent = ESTRLowOFUL(
            d1=d_x,
            d2=d_y,
            r=getattr(cfg, 'estr_r', min(d_x, d_y)),
            T1=getattr(cfg, 'estr_T1', M * N),
            lam=getattr(cfg, 'estr_lam', cfg.p),
            lam_perp=getattr(cfg, 'estr_lam_perp', cfg.p),
            B=getattr(cfg, 'estr_B', 1.0),
            B_perp=getattr(cfg, 'estr_B_perp', 1.0),
            delta=cfg.delta,
            sigma=noise_std,
        )
    
    ## sample features
    ## X_star: (d_x_star, M)
    ## X: (d_x, M)
    ## Y_star: (d_y_star, N)
    ## Y: (d_y, N)
    X_star, X, Y_star, Y = bilinear_feature_generator(
        case=case,
        d_x_star=d_x_star,
        d_x=d_x,
        d_y_star=d_y_star,
        d_y=d_y,
        M=M,
        N=N,
    )

    # ## Z_star: (d_x_star * d_y_star, MN): col = (1,1), (1,2), ... , (1,N), (2,1),  ... , ... , (M,1), (M,2), ... , (M,N)
    # ## Z: (d_x * d_y, MN): col = (1,1), (1,2), ... , (1,N), (2,1),  ... , ... , (M,1), (M,2), ... , (M,N)
    # Z_star, Z = np.kron(X_star, Y_star), np.kron(X, Y)

    ## sample reward parameter after augmentation and compute the expected rewards
    reward_param_mat = bilinear_param_generator(
        dimension_x=d_x_star,
        dimension_y=d_y_star,
        distribution=cfg.param_dist,
        disjoint=True,
        bound=cfg.param_bound,
        bound_type=cfg.param_bound_type,
        uniform_rng=cfg.param_uniform_rng,
    )

    ## (M,N) matrix with the maximum absolute value does not exceed 1
    exp_rewards_mat = X_star.T @ reward_param_mat @ Y_star
    exp_rewards_mat = exp_rewards_mat / np.max(np.abs(exp_rewards_mat))

    if (
        isinstance(agent, (LinUCB,LinTS,DRLassoBandit,ESTRLowOFUL))
    ):
        data_x = X.T
        data_y = Y.T
    else:
        # (M, M-d) matrix and each column vector denotes the orthogonal basis if M > d
        # (M, M) matrix from singular value decomposition if d > M
        basis_X = orthogonal_complement_basis(X)

        # (N, N-d) matrix and each column vector denotes the orthogonal basis if N > d
        # (N, N) matrix from singular value decomposition if d > N
        basis_Y = orthogonal_complement_basis(Y)

        d_X, M = X.shape
        if d_X <= M:
            x_aug = np.hstack(
                (X.T, basis_X)
            )  # augmented into (M, M) matrix and each row vector denotes the augmented feature
            data_x = x_aug
        else:
            data_x = basis_X

        d_Y, N = Y.shape
        if d_Y <= N:
            y_aug = np.hstack(
                (Y.T, basis_Y)
            )  # augmented into (N, N) matrix and each row vector denotes the augmented feature
            data_y = y_aug
        else:
            data_y = basis_Y

    # print(f"Agent : {agent.__class__.__name__}\t data shape : {data.shape}")

    # Set timing data for BiRoLF agents
    if hasattr(agent, '__class__') and agent.__class__.__name__ in ['RoLFLasso', 'BiRoLFLasso', 'BiRoLFLasso_FISTA', 'BiRoLFLasso_Blockwise']:
        agent._timing_data = timing_data
        agent._trial = now_trial

    # Measure total execution time
    trial_start_time = time.time()
    regrets = bilinear_run(
        trial=now_trial,
        agent=agent,
        horizon=horizon,
        exp_rewards_mat=exp_rewards_mat,
        x=data_x,
        y=data_y,
        noise_dist=cfg.reward_dist,
        noise_std=noise_std,
        verbose=verbose,
        fname=fname,
        timing_data=timing_data,
    )
    trial_total_time = time.time() - trial_start_time
    
    # Store total execution time in local timing data
    agent_name = agent.__class__.__name__
    if timing_data is not None:
        if 'total_execution_times' not in timing_data:
            timing_data['total_execution_times'] = {}
        if agent_name not in timing_data['total_execution_times']:
            timing_data['total_execution_times'][agent_name] = []
        timing_data['total_execution_times'][agent_name].append(trial_total_time)

    regret_container[0] = regrets
    return regret_container


## Each data is augmented when using RoLF-like algorithm
def bilinear_run(
    trial: int,
    agent: Union[MAB, ContextualBandit],
    horizon: int,
    exp_rewards_mat: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    noise_dist: str,
    noise_std: float,
    verbose: bool,
    fname: str,
    timing_data: dict = None,
):
    # x, y: augmented feature if the agent is RoLF (M, M), (N, N) each.
    regrets = np.zeros(horizon, dtype=float)

    if not verbose:
        bar = tqdm(range(horizon))
    else:
        bar = range(horizon)

    # For linear contextual bandits
    # For RoLF this is (MN,MN), otherwise (MN,d_x*d_y)
    z = None
    if not isinstance(agent, (BiRoLFLasso, BiRoLFLasso_FISTA, BiRoLFLasso_Blockwise, ESTRLowOFUL)):
        z = np.kron(x, y)

    # z = np.kron(x, y)

    for t in bar:
        # if t == 0:
        #     print(f"Number of actions : {x.shape[0]}\tReward range : [{np.amin(exp_rewards):.5f}, {np.amax(exp_rewards):.5f}]")

        ## compute the optimal action
        optimal_action = np.argmax(exp_rewards_mat)
        M, N = exp_rewards_mat.shape
        optimal_i, optimal_j = action_to_ij(optimal_action, N)
        optimal_reward = exp_rewards_mat[optimal_i][optimal_j]

        ## choose the best action
        noise = subgaussian_noise(
            distribution=noise_dist, size=1, std=noise_std
        )

        if isinstance(agent, (BiRoLFLasso, BiRoLFLasso_FISTA, BiRoLFLasso_Blockwise, ESTRLowOFUL)):
            chosen_action = agent.choose(x, y)
        elif isinstance(agent, ContextualBandit):
            chosen_action = agent.choose(z)
        else:
            chosen_action = agent.choose()
        chosen_i, chosen_j = action_to_ij(chosen_action, N)
        chosen_reward = exp_rewards_mat[chosen_i][chosen_j] + noise

        # HERE
        if t % 10 == 0  and verbose:
            try:
                string = f"""
                        case : {cfg.case}, SEED : {cfg.seed}, M : {cfg.arm_x}, N: {cfg.arm_y},
                        true_dim_x : {cfg.true_dim_x}, true_dim_y : {cfg.true_dim_y}, Obs_dim_x : {cfg.dim_x}, Obs_dim_y : {cfg.dim_y},
                        Trial : {trial}, p : {cfg.p}, Agent : {agent.__class__.__name__},
                        Round : {t+1}, optimal : {optimal_action}, a_hat: {agent.a_hat},
                        pseudo : {agent.pseudo_action}, chosen action : {ij_to_action(chosen_i,chosen_j,cfg.arm_y)}
                    """
            except:
                string = f"""
                        case : {cfg.case}, SEED : {cfg.seed}, M : {cfg.arm_x}, N: {cfg.arm_y},
                        true_dim_x : {cfg.true_dim_x}, true_dim_y : {cfg.true_dim_y}, Obs_dim_x : {cfg.dim_x}, Obs_dim_y : {cfg.dim_y},
                        Trial : {trial}, p : {cfg.p}, Agent : {agent.__class__.__name__},
                        Round : {t+1}, optimal : {optimal_action}, chosen action: {ij_to_action(chosen_i,chosen_j,cfg.arm_y)}
                    """
                
            save_log(path=LOG_PATH, fname=fname, string=" ".join(string.split()))
            print(" ".join(string.split()))

        ## compute the regret
        regrets[t] = optimal_reward - exp_rewards_mat[chosen_i, chosen_j]

        ## update the agent with timing measurement
        update_start_time = time.time()
        if isinstance(agent, (BiRoLFLasso, BiRoLFLasso_FISTA, BiRoLFLasso_Blockwise, ESTRLowOFUL)):
            agent.update(x=x, y=y, r=chosen_reward)
        elif isinstance(agent, ContextualBandit):
            agent.update(x=z, r=chosen_reward)
        else:
            agent.update(a=chosen_action, r=chosen_reward)
        update_time = time.time() - update_start_time
        
        # Store total update time (not just lasso optimization time)
        # The lasso optimization timing is handled separately in models.py

    return np.cumsum(regrets)


## nothing change compare to show_result()
def bilinear_show_result(
    regrets: dict, horizon: int, figsize: tuple = (6, 5), fontsize=11
):
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["blue", "orange", "green", "red", "purple", "black", "brown", "olive", "cyan"]
    period = horizon // 10

    z_init = len(colors)
    # Plot the graph for each algorithm with error bars
    for i, (color, (key, item)) in enumerate(zip(colors, regrets.items())):
        rounds = np.arange(horizon)
        mean = np.mean(item, axis=0)
        std = np.std(item, axis=0, ddof=1)

        # Display the line with markers and error bars periodically
        ax.errorbar(
            rounds[::period],
            mean[::period],
            yerr=std[::period],
            label=f"{key}",
            fmt="s",
            color=color,
            capsize=3,
            elinewidth=1,
            zorder=z_init - i,
        )

        # Display the full line without periodic markers
        ax.plot(rounds, mean, color=color, linewidth=2, zorder=z_init - i)

    ax.grid(True)
    ax.set_xlabel(r"Round ($t$)")
    ax.set_ylabel("Cumulative Regret")
    ax.legend(loc="upper left", fontsize=fontsize)

    fig.tight_layout()
    return fig


# Function to run trials for a single agent
def bilinear_run_agent(args):
    trial_agent, shared_timing_data, shared_total_times = args
    now_trial, agent_type = trial_agent
    _maybe_set_blas_threads()
    np.random.seed(cfg.seed + (513 * now_trial))

    start = time.perf_counter()

    # Initialize local timing data for this process
    local_timing_data = {}
    
    regrets = bilinear_run_trial(
        agent_type=agent_type,
        now_trial=now_trial,
        horizon=cfg.horizon,
        d_x_star=cfg.true_dim_x,
        d_x=cfg.dim_x,
        M=cfg.arm_x,
        d_y_star=cfg.true_dim_y,
        d_y=cfg.dim_y,
        N=cfg.arm_y,
        noise_std=cfg.reward_std,
        case=cfg.case,
        verbose=True,
        fname=f"Case_{cfg.case}_Agent_{agent_type}_M_{cfg.arm_x}_N_{cfg.arm_y}_xstar_{cfg.true_dim_x}_ystar_{cfg.true_dim_y}_dx_{cfg.dim_x}_dy_{cfg.dim_y}_T_{cfg.horizon}_explored_{cfg.init_explore}_noise_{cfg.reward_std}_run_{RUN_TAG}",
        timing_data=local_timing_data  # Pass timing data container
    )
    
    end = time.perf_counter()
    
    # Save timing data to temporary file for this process
    agent_display_name = AGENT_DICT[agent_type]
    # Create unique filename including experiment parameters to avoid collisions
    timing_file = f"/tmp/timing_{agent_type}_{now_trial}_M_{cfg.arm_x}_N_{cfg.arm_y}_xstar_{cfg.true_dim_x}_ystar_{cfg.true_dim_y}_dx_{cfg.dim_x}_dy_{cfg.dim_y}_T_{cfg.horizon}_noise_{cfg.reward_std}_run_{RUN_TAG}.pkl"
    timing_info = {
        'optimization_times': local_timing_data,
        'total_time': end - start,
        'agent_name': agent_display_name,
        'trial': now_trial
    }
    
    try:
        with open(timing_file, 'wb') as f:
            pickle.dump(timing_info, f)
    except Exception as e:
        print(f"Warning: Could not save timing data: {e}")

    key = (now_trial, agent_display_name)
    return key, regrets, end - start, timing_file


def plot_optimization_timing_comparison():
    """Plot average optimization time comparison for RoLFLasso and BiRoLFLasso variants"""
    if not TIMING_DATA:
        print("No timing data available for optimization comparison.")
        return
    
    # Calculate average optimization times
    avg_times = {}
    std_times = {}

    agent_order = [
        "RoLFLasso",
        "BiRoLFLasso",
        "BiRoLFLasso_FISTA",
        "BiRoLFLasso_Blockwise",
    ]
    display_names = {
        "RoLFLasso": "RoLF-Lasso",
        "BiRoLFLasso": "BiRoLF-Lasso",
        "BiRoLFLasso_FISTA": "BiRoLF-Lasso-FISTA",
        "BiRoLFLasso_Blockwise": "BiRoLF-Lasso-Blockwise",
    }

    for agent_name in agent_order:
        if agent_name in TIMING_DATA:
            all_times = []
            for trial_times in TIMING_DATA[agent_name].values():
                all_times.extend(trial_times)
            if all_times:
                avg_times[agent_name] = np.mean(all_times)
                std_times[agent_name] = np.std(all_times)

    if not avg_times:
        print("No optimization timing data to plot.")
        return
    
    # Create bar plot with enhanced visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    agents = list(avg_times.keys())
    times = list(avg_times.values())
    errors = [std_times[agent] for agent in agents]
    
    # Create bars with different colors and transparency for variance
    palette = {
        "RoLFLasso": "#6B7280",
        "BiRoLFLasso": "#4472C4",
        "BiRoLFLasso_FISTA": "#E15759",
        "BiRoLFLasso_Blockwise": "#59A14F",
    }
    colors = [palette.get(agent, "#A5A5A5") for agent in agents]
    x_pos = np.arange(len(agents))
    
    # Main bars
    bars = ax.bar(x_pos, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)
    
    # Add error bars (standard deviation)
    ax.errorbar(x_pos, times, yerr=errors, fmt='none', color='black', capsize=8, capthick=2, linewidth=2)
    
    # Add shaded regions for standard deviation
    for i, (x, time_val, err) in enumerate(zip(x_pos, times, errors)):
        ax.fill_between([x-0.3, x+0.3], [time_val-err, time_val-err], [time_val+err, time_val+err], 
                       color=colors[i], alpha=0.2, label=f'±1σ {agents[i]}' if i < 2 else None)
    
    # Add value labels on bars
    for i, (bar, time_val, err) in enumerate(zip(bars, times, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err + max(times) * 0.02,
                f'{time_val:.6f}s\n±{err:.6f}s', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Average Optimization Time (seconds)', fontsize=14, fontweight='bold')
    title_params = (
        f"Case_{cfg.case}_M_{cfg.arm_x}_N_{cfg.arm_y}_xstar_{cfg.true_dim_x}_ystar_{cfg.true_dim_y}_"
        f"dx_{cfg.dim_x}_dy_{cfg.dim_y}_T_{cfg.horizon}_explored_{cfg.init_explore}_noise_{cfg.reward_std}_run_{RUN_TAG}"
    )
    ax.set_title(f'Optimization Time Comparison\n{title_params}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Make agent names more readable
    ax.set_xticks(x_pos)
    ax.set_xticklabels([display_names.get(agent, agent) for agent in agents], fontsize=12)
    
    # Add legend for variance shading
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    # Generate filename with experiment parameters
    fname_params = f"M_{cfg.arm_x}_N_{cfg.arm_y}_T_{cfg.horizon}_trials_{cfg.trials}_seed_{cfg.seed}"
    
    # Save plot
    os.makedirs(FIGURE_PATH, exist_ok=True)
    plt.savefig(f"{FIGURE_PATH}/optimization_timing_comparison_{fname_params}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{FIGURE_PATH}/optimization_timing_comparison_{fname_params}.pdf", bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION TIMING COMPARISON SUMMARY")
    print("="*60)
    for agent_name in agents:
        print(f"{agent_name}: {avg_times[agent_name]:.6f} ± {std_times[agent_name]:.6f} seconds")
    print("="*60)


def plot_total_execution_time_comparison():
    """Plot total execution time comparison across all algorithms"""
    if not TOTAL_EXECUTION_TIMES:
        print("No total execution time data available.")
        return
    
    # Calculate statistics
    avg_times = {}
    std_times = {}
    
    for agent_name, times in TOTAL_EXECUTION_TIMES.items():
        if times:
            avg_times[agent_name] = np.mean(times)
            std_times[agent_name] = np.std(times)
    
    if not avg_times:
        print("No execution time data to plot.")
        return
    
    # Sort by average time for better visualization
    sorted_agents = sorted(avg_times.keys(), key=lambda x: avg_times[x])
    
    # Create horizontal bar plot for better readability
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y_pos = np.arange(len(sorted_agents))
    times = [avg_times[agent] for agent in sorted_agents]
    errors = [std_times[agent] for agent in sorted_agents]
    
    # Color BiRoLF algorithms differently with better color scheme
    colors = []
    edge_colors = []
    for agent in sorted_agents:
        if 'BiRoLF' in agent:
            if 'FISTA' in agent:
                colors.append('#E15759')  # Red for FISTA
                edge_colors.append('#B91C1C')
            else:
                colors.append('#4472C4')  # Blue for original BiRoLF
                edge_colors.append('#1E40AF')
        else:
            colors.append('#A5A5A5')  # Gray for others
            edge_colors.append('#6B7280')
    
    # Main bars
    bars = ax.barh(y_pos, times, color=colors, alpha=0.8, edgecolor=edge_colors, linewidth=1.5, height=0.6)
    
    # Add error bars (standard deviation)
    ax.errorbar(times, y_pos, xerr=errors, fmt='none', color='black', capsize=5, capthick=1.5, linewidth=1.5)
    
    # Add shaded regions for standard deviation
    for i, (y, time_val, err, color) in enumerate(zip(y_pos, times, errors, colors)):
        ax.fill_betweenx([y-0.3, y+0.3], [time_val-err, time_val-err], [time_val+err, time_val+err], 
                        color=color, alpha=0.2)
    
    # Add value labels with improved formatting
    for i, (bar, time_val, err) in enumerate(zip(bars, times, errors)):
        width = bar.get_width()
        ax.text(width + err + max(times) * 0.02, bar.get_y() + bar.get_height()/2.,
                f'{time_val:.3f}s ± {err:.3f}s', ha='left', va='center', 
                fontweight='bold', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([AGENT_DICT.get(agent, agent) for agent in sorted_agents], fontsize=11)
    ax.set_xlabel('Total Execution Time per Trial (seconds)', fontsize=14, fontweight='bold')
    title_params = (
        f"Case_{cfg.case}_M_{cfg.arm_x}_N_{cfg.arm_y}_xstar_{cfg.true_dim_x}_ystar_{cfg.true_dim_y}_"
        f"dx_{cfg.dim_x}_dy_{cfg.dim_y}_T_{cfg.horizon}_explored_{cfg.init_explore}_noise_{cfg.reward_std}_run_{RUN_TAG}"
    )
    ax.set_title(f'Total Execution Time Comparison\n{title_params}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Highlight BiRoLF algorithms
    for i, agent in enumerate(sorted_agents):
        if 'BiRoLF' in agent:
            ax.get_yticklabels()[i].set_weight('bold')
            if 'FISTA' in agent:
                ax.get_yticklabels()[i].set_color('#E15759')
            else:
                ax.get_yticklabels()[i].set_color('#4472C4')
    
    plt.tight_layout()
    
    # Generate filename with experiment parameters
    fname_params = f"M_{cfg.arm_x}_N_{cfg.arm_y}_T_{cfg.horizon}_trials_{cfg.trials}_seed_{cfg.seed}"
    
    # Save plot
    os.makedirs(FIGURE_PATH, exist_ok=True)
    plt.savefig(f"{FIGURE_PATH}/total_execution_time_comparison_{fname_params}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{FIGURE_PATH}/total_execution_time_comparison_{fname_params}.pdf", bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("TOTAL EXECUTION TIME COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<25} {'Avg Time (s)':<15} {'Std Time (s)':<15} {'Trials':<10}")
    print("-"*80)
    for agent_name in sorted_agents:
        agent_display = AGENT_DICT.get(agent_name, agent_name)
        n_trials = len(TOTAL_EXECUTION_TIMES[agent_name])
        print(f"{agent_display:<25} {avg_times[agent_name]:<15.3f} {std_times[agent_name]:<15.3f} {n_trials:<10}")
    print("="*80)


def save_timing_data():
    """Save timing data to pkl files"""
    import pickle
    import os
    
    # Create results directory
    os.makedirs(RESULT_PATH, exist_ok=True)
    
    # Check if timing data exists
    if TIMING_DATA is None or TOTAL_EXECUTION_TIMES is None:
        print("No timing data available to save.")
        return
    
    # Prepare timing data for saving
    timing_summary = {
        'optimization_times': dict(TIMING_DATA) if TIMING_DATA else {},
        'total_execution_times': dict(TOTAL_EXECUTION_TIMES) if TOTAL_EXECUTION_TIMES else {},
        'config': vars(cfg),
        'timestamp': dt.now().isoformat()
    }
    
    # Calculate statistics
    timing_stats = {}
    
    # Optimization timing statistics
    for agent_name in TIMING_DATA:
        all_times = []
        for trial_times in TIMING_DATA[agent_name].values():
            all_times.extend(trial_times)
        
        if all_times:
            timing_stats[agent_name] = {
                'optimization': {
                    'mean': np.mean(all_times),
                    'std': np.std(all_times),
                    'min': np.min(all_times),
                    'max': np.max(all_times),
                    'count': len(all_times)
                }
            }
    
    # Total execution timing statistics
    for agent_name in TOTAL_EXECUTION_TIMES:
        times = TOTAL_EXECUTION_TIMES[agent_name]
        if times:
            if agent_name not in timing_stats:
                timing_stats[agent_name] = {}
            timing_stats[agent_name]['total_execution'] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times)
            }
    
    timing_summary['statistics'] = timing_stats
    
    # Generate filename
    case = cfg.case
    M, N = cfg.arm_x, cfg.arm_y
    d_x_star, d_y_star = cfg.true_dim_x, cfg.true_dim_y
    d_x, d_y = cfg.dim_x, cfg.dim_y
    T = cfg.horizon
    sigma = cfg.reward_std
    
    timing_fname = f"timing_Case_{case}_M_{M}_N_{N}_xstar_{d_x_star}_ystar_{d_y_star}_dx_{d_x}_dy_{d_y}_T_{T}_explored_{cfg.init_explore}_noise_{sigma}_run_{RUN_TAG}"
    
    # Save detailed timing data
    with open(f"{RESULT_PATH}/{timing_fname}_detailed.pkl", "wb") as f:
        pickle.dump(timing_summary, f)
    
    # Save just statistics for easy loading
    with open(f"{RESULT_PATH}/{timing_fname}_stats.pkl", "wb") as f:
        pickle.dump(timing_stats, f)
    
    print(f"Timing data saved to:")
    print(f"  - {RESULT_PATH}/{timing_fname}_detailed.pkl")
    print(f"  - {RESULT_PATH}/{timing_fname}_stats.pkl")


def plot_timing_analysis():
    """Generate all timing analysis plots and save timing data"""
    print("Generating timing analysis plots...")
    plot_optimization_timing_comparison()
    plot_total_execution_time_comparison()
    
    # Save timing data as pkl file
    save_timing_data()
    print("Timing analysis plots and data saved successfully!")


if __name__ == "__main__":
    ##

    ## hyper-parameters
    M = cfg.arm_x
    N = cfg.arm_y

    d_x_star = cfg.true_dim_x
    d_y_star = cfg.true_dim_y

    d_x = cfg.dim_x
    d_y = cfg.dim_y

    T = cfg.horizon
    SEED = cfg.seed
    sigma = cfg.reward_std
    AGENTS = [
        # "birolf_lasso",
        "birolf_lasso_fista",
        "birolf_lasso_blockwise",
        "rolf_lasso",
        "rolf_ridge",
        "dr_lasso",
        "linucb",
        "lints",
        "mab_ucb",
    ]
    TRIALS_AGENTS = []
    for _agent in AGENTS:
        for _trial in range(cfg.trials):
            TRIALS_AGENTS.append((_trial,_agent))
    
    case = cfg.case

    # Parallel execution using ProcessPoolExecutor
    # Prepare arguments for each worker (simplified - no shared data needed)
    worker_args = [(trial_agent, None, None) for trial_agent in TRIALS_AGENTS]
    
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = executor.map(bilinear_run_agent, worker_args)

    # Collect results and timing data
    regret_results = dict()
    time_check = dict()
    timing_files = []
    
    for key, regrets, time, timing_file in results:
        now_trial, agent_type = key

        if agent_type not in regret_results.keys():
            regret_results[agent_type] = np.zeros(cfg.trials, dtype=object)
            time_check[agent_type] = time

        regret_results[agent_type][now_trial] = regrets[0]
        time_check[agent_type] += time
        
        if timing_file:
            timing_files.append(timing_file)

    # Load timing data from temporary files
    for timing_file in timing_files:
        try:
            with open(timing_file, 'rb') as f:
                timing_info = pickle.load(f)
            
            # Process optimization timing data
            if timing_info['optimization_times']:
                for class_name, trial_data in timing_info['optimization_times'].items():
                    if class_name not in TIMING_DATA:
                        TIMING_DATA[class_name] = {}
                    for trial_num, times in trial_data.items():
                        if trial_num not in TIMING_DATA[class_name]:
                            TIMING_DATA[class_name][trial_num] = []
                        TIMING_DATA[class_name][trial_num].extend(times)
            
            # Process total execution timing data
            agent_name = timing_info['agent_name']
            trial_num = timing_info['trial']
            total_time = timing_info['total_time']
            
            if agent_name not in TOTAL_EXECUTION_TIMES:
                TOTAL_EXECUTION_TIMES[agent_name] = []
            # Ensure list is long enough
            while len(TOTAL_EXECUTION_TIMES[agent_name]) <= trial_num:
                TOTAL_EXECUTION_TIMES[agent_name].append(0)
            TOTAL_EXECUTION_TIMES[agent_name][trial_num] = total_time
            
            # Clean up temporary file
            os.remove(timing_file)
            
        except Exception as e:
            print(f"Warning: Could not load timing data from {timing_file}: {e}")

    fig = bilinear_show_result(regrets=regret_results, horizon=T, fontsize=15)

    fname = f"Case_{case}_M_{M}_N_{N}_xstar_{d_x_star}_ystar_{d_y_star}_dx_{d_x}_dy_{d_y}_T_{T}_explored_{cfg.init_explore}_noise_{sigma}_run_{RUN_TAG}"

    save_plot(fig, path=FIGURE_PATH, time_check = time_check, fname=fname)
    save_result(
        result=(vars(cfg), regret_results),
        time_check = time_check,
        path=RESULT_PATH,
        fname=fname,
        filetype=cfg.filetype,
    )
    
    # Generate timing analysis plots
    print("\n" + "="*80)
    print("GENERATING TIMING ANALYSIS PLOTS")
    print("="*80)
    plot_timing_analysis()
    print("All plots and results saved successfully!")

#
###### ! BEFORE CHANGE ! #############
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


# from cfg import get_cfg
# from models import *
# from util import *

# MOTHER_PATH = "."

# DIST_DICT = {"gaussian": "g", "uniform": "u"}

# AGENT_DICT = {
#     "mab_ucb": r"UCB($\delta$)",
#     "linucb": "LinUCB",
#     "lints": "LinTS",
#     "rolf_lasso": "RoLF-Lasso (Ours)",
#     "rolf_ridge": "RoLF-Ridge (Ours)",
#     "dr_lasso": "DRLasso",
# }

# cfg = get_cfg()

# RESULT_PATH = f"{MOTHER_PATH}/results/{str(cfg.date)}/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"
# FIGURE_PATH = f"{MOTHER_PATH}/figures/{str(cfg.date)}/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"
# LOG_PATH = (
#     f"{MOTHER_PATH}/logs/{str(cfg.date)}/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"
# )

# ## ~! Generate feature matrix Z(full feature), X(observerable feature) !~


# def feature_generator(case: int, d_z: int, d: int, K: int, random_state: int):
#     ## sample the true, observable, and unobservable features
#     d_u = d_z - d  # dimension of unobserved features
#     assert case in [1, 2, 3], "There exists only Case 1, 2, and 3."
#     if case == 1:
#         ## Default case
#         np.random.seed(random_state)
#         Z = np.random.multivariate_normal(
#             mean=np.zeros(d_z), cov=np.eye(d_z), size=K
#         ).T  # (k, K)
#         X = Z[:d, :]  # (d, K)

#     # For two matrices A and B,
#     # if each row of A can be expressed as a linear combination of the rows of B,
#     # then R(A) ⊆ R(B)
#     elif case == 2:

#         ## ~! When observable feature dominates !~

#         ## R(U) ⊆ R(X)
#         np.random.seed(random_state + 17)
#         # First generate X
#         X = np.random.multivariate_normal(
#             mean=np.zeros(d), cov=np.eye(d), size=K
#         ).T  # (d, K)

#         # Generate a coefficient matrix C
#         # C = np.random.multivariate_normal(mean=np.zeros(d),
#         #                                   cov=np.eye(d),
#         #                                   size=d_u).T # (d_u, d)
#         C = np.random.uniform(low=-1 / np.pi, high=1 / np.pi, size=(d_u, d))  # (d_u, d)

#         # Compute U as a multiplication between C and X
#         U = C @ X  # (d_u, K)
#         Z = np.concatenate([X, U], axis=0)  # (k, K)))

#     elif case == 3:

#         ## ~! When the unobservable feature dominates !~

#         ## R(X) ⊆ R(U)
#         np.random.seed(random_state + 31)
#         # First generate U
#         U = np.random.multivariate_normal(
#             mean=np.zeros(d_u), cov=np.eye(d_u), size=K
#         ).T  # (d_u, K)

#         # Generate a coefficient matrix C
#         # C = np.random.multivariate_normal(mean=np.zeros(d),
#         #                                   cov=np.eye(d),
#         #                                   size=d_u).T # (d, d_u)
#         C = np.random.uniform(low=-1 / np.pi, high=1 / np.pi, size=(d, d_u))  # (d, d_u)

#         # Compute U as a multiplication between C and X
#         X = C @ U  # (d, K)
#         Z = np.concatenate([X, U], axis=0)  # (k, K)))

#     return Z, X


# def run_trials(
#     agent_type: str,
#     trials: int,
#     horizon: int,
#     k: int,
#     d: int,
#     arms: int,
#     noise_std: float,
#     case: int,
#     random_state: int,
#     verbose: bool,
#     fname: str,
# ):

#     exp_map = {
#         "double": (2 * arms),
#         "sqr": (arms**2),
#         "K": arms,
#         "triple": (3 * arms),
#         "quad": (4 * arms),
#     }

#     ## run and collect the regrets
#     regret_container = np.zeros(trials, dtype=object)
#     for trial in range(trials):
#         if random_state is not None:
#             random_state_ = random_state + (513 * trial)
#         else:
#             random_state_ = None

#         if agent_type == "linucb":
#             agent = LinUCB(d=d, lbda=cfg.p, delta=cfg.delta)

#         elif agent_type == "lints":
#             agent = LinTS(
#                 d=d, lbda=cfg.p, horizon=horizon, reward_std=noise_std, delta=cfg.delta
#             )

#         elif agent_type == "mab_ucb":
#             agent = UCBDelta(n_arms=arms, delta=cfg.delta)

#         elif agent_type == "rolf_lasso":
#             if cfg.explore:
#                 agent = RoLFLasso(
#                     d=d,
#                     arms=arms,
#                     p=cfg.p,
#                     delta=cfg.delta,
#                     sigma=noise_std,
#                     random_state=random_state_,
#                     explore=cfg.explore,
#                     init_explore=exp_map[cfg.init_explore],
#,
                # lam_c_impute=lam_c_impute_lin, lam_c_main=lam_c_main_lin)
#             else:
#                 agent = RoLFLasso(
#                     d=d,
#                     arms=arms,
#                     p=cfg.p,
#                     delta=cfg.delta,
#                     sigma=noise_std,
#                     random_state=random_state_,
#,
                # lam_c_impute=lam_c_impute_lin, lam_c_main=lam_c_main_lin)

#         elif agent_type == "rolf_ridge":
#             if cfg.explore:
#                 agent = RoLFRidge(
#                     d=d,
#                     arms=arms,
#                     p=cfg.p,
#                     delta=cfg.delta,
#                     sigma=noise_std,
#                     random_state=random_state_,
#                     explore=cfg.explore,
#                     init_explore=exp_map[cfg.init_explore],
#                 )
#             else:
#                 agent = RoLFRidge(
#                     d=d,
#                     arms=arms,
#                     p=cfg.p,
#                     delta=cfg.delta,
#                     sigma=noise_std,
#                     random_state=random_state_,
#                 )

#         elif agent_type == "dr_lasso":
#             agent = DRLassoBandit(d=d, arms=arms, lam1=1.0, lam2=0.5, zT=10, tr=True)

#         ## sample features
#         Z, X = feature_generator(
#             case=case, d_z=k, d=d, K=arms, random_state=random_state_ + 1
#         )

#         ## sample reward parameter after augmentation and compute the expected rewards
#         reward_param = param_generator(
#             dimension=k,
#             distribution=cfg.param_dist,
#             disjoint=cfg.param_disjoint,
#             bound=cfg.param_bound,
#             bound_type=cfg.param_bound_type,
#             uniform_rng=cfg.param_uniform_rng,
#             random_state=random_state_,
#         )

#         ## (K, ) vector with the maximum absolute value does not exceed 1
#         exp_rewards = bounding(
#             type="param", v=Z.T @ reward_param, bound=1.0, norm_type="lsup"
#         )

#         if (
#             isinstance(agent, LinUCB)
#             or isinstance(agent, LinTS)
#             or isinstance(agent, DRLassoBandit)
#         ):
#             data = X.T  # (K, d)
#         else:
#             # (K, K-d) matrix and each column vector denotes the orthogonal basis if K > d
#             # (K, K) matrix from singular value decomposition if d > K
#             basis = orthogonal_complement_basis(X)

#             d, K = X.shape
#             if d <= K:
#                 x_aug = np.hstack(
#                     (X.T, basis)
#                 )  # augmented into (K, K) matrix and each row vector denotes the augmented feature
#                 data = x_aug
#             else:
#                 data = basis

#         # print(f"Agent : {agent.__class__.__name__}\t data shape : {data.shape}")

#         regrets = run(
#             trial=trial,
#             agent=agent,
#             horizon=horizon,
#             exp_rewards=exp_rewards,
#             x=data,
#             noise_dist=cfg.reward_dist,
#             noise_std=noise_std,
#             random_state=random_state_,
#             verbose=verbose,
#             fname=fname,
#         )

#         regret_container[trial] = regrets
#     return regret_container


# def run(
#     trial: int,
#     agent: Union[MAB, ContextualBandit],
#     horizon: int,
#     exp_rewards: np.ndarray,
#     x: np.ndarray,
#     noise_dist: str,
#     noise_std: float,
#     random_state: int,
#     verbose: bool,
#     fname: str,
# ):

#     # x: augmented feature if the agent is RoLF (K, K)
#     regrets = np.zeros(horizon, dtype=float)

#     if not verbose:
#         bar = tqdm(range(horizon))
#     else:
#         bar = range(horizon)

#     for t in bar:
#         if random_state is not None:
#             random_state_ = random_state + int(113 * t)
#         else:
#             random_state_ = None

#         # if t == 0:
#         #     print(f"Number of actions : {x.shape[0]}\tReward range : [{np.amin(exp_rewards):.5f}, {np.amax(exp_rewards):.5f}]")

#         ## compute the optimal action
#         optimal_action = np.argmax(exp_rewards)
#         optimal_reward = exp_rewards[optimal_action]

#         ## choose the best action
#         noise = subgaussian_noise(
#             distribution=noise_dist, size=1, std=noise_std, random_state=random_state_
#         )

#         if isinstance(agent, ContextualBandit):
#             chosen_action = agent.choose(x)
#         else:
#             chosen_action = agent.choose()
#         chosen_reward = exp_rewards[chosen_action] + noise

#         if verbose:
#             try:
#                 string = f"""
#                         case : {cfg.case}, SEED : {cfg.seed}, K : {cfg.arms},
#                         Latent_dim : {cfg.latent_dim}, Obs_dim : {cfg.dim},
#                         Trial : {trial}, p : {cfg.p}, Agent : {agent.__class__.__name__},
#                         Round : {t+1}, optimal : {optimal_action}, a_hat: {agent.a_hat},
#                         pseudo : {agent.pseudo_action}, chosen : {agent.chosen_action}
#                     """
#             except:
#                 string = f"""
#                         case : {cfg.case}, SEED : {cfg.seed}, K : {cfg.arms},
#                         Latent_dim : {cfg.latent_dim}, Obs_dim : {cfg.dim},
#                         Trial : {trial}, p : {cfg.p}, Agent : {agent.__class__.__name__},
#                         Round : {t+1}, optimal : {optimal_action}, chosen : {chosen_action}
#                     """
#             save_log(path=LOG_PATH, fname=fname, string=" ".join(string.split()))
#             print(" ".join(string.split()))

#         ## compute the regret
#         regrets[t] = optimal_reward - exp_rewards[chosen_action]

#         ## update the agent
#         if isinstance(agent, ContextualBandit):
#             agent.update(x=x, r=chosen_reward)
#         else:
#             agent.update(a=chosen_action, r=chosen_reward)

#     return np.cumsum(regrets)


# def show_result(regrets: dict, horizon: int, figsize: tuple = (6, 5), fontsize=11):

#     fig, ax = plt.subplots(figsize=figsize)

#     colors = ["blue", "orange", "green", "red", "purple", "black"]
#     period = horizon // 10

#     z_init = len(colors)
#     # Plot the graph for each algorithm with error bars
#     for i, (color, (key, item)) in enumerate(zip(colors, regrets.items())):
#         rounds = np.arange(horizon)
#         mean = np.mean(item, axis=0)
#         std = np.std(item, axis=0, ddof=1)

#         # Display the line with markers and error bars periodically
#         ax.errorbar(
#             rounds[::period],
#             mean[::period],
#             yerr=std[::period],
#             label=f"{key}",
#             fmt="s",
#             color=color,
#             capsize=3,
#             elinewidth=1,
#             zorder=z_init - i,
#         )

#         # Display the full line without periodic markers
#         ax.plot(rounds, mean, color=color, linewidth=2, zorder=z_init - i)

#     ax.grid(True)
#     ax.set_xlabel(r"Round ($t$)")
#     ax.set_ylabel("Cumulative Regret")
#     ax.legend(loc="upper left", fontsize=fontsize)

#     fig.tight_layout()
#     return fig


# # Function to run trials for a single agent
# def run_agent(agent_type):
#     regrets = run_trials(
#         agent_type=agent_type,
#         trials=cfg.trials,
#         horizon=cfg.horizon,
#         k=cfg.latent_dim,
#         d=cfg.dim,
#         arms=cfg.arms,
#         noise_std=cfg.reward_std,
#         case=cfg.case,
#         random_state=cfg.seed,
#         verbose=True,
#         fname=f"Case_{cfg.case}_K_{cfg.arms}_k_{cfg.latent_dim}_d_{cfg.dim}_T_{cfg.horizon}_explored_{cfg.init_explore}_noise_{cfg.reward_std}",
#     )
#     key = AGENT_DICT[agent_type]
#     return key, regrets


# if __name__ == "__main__":
#     ## hyper-parameters
#     arms = cfg.arms  # List[int]
#     k = cfg.latent_dim
#     d = cfg.dim
#     T = cfg.horizon
#     SEED = cfg.seed
#     sigma = cfg.reward_std
#     AGENTS = ["rolf_lasso", "rolf_ridge", "dr_lasso", "linucb", "lints", "mab_ucb"]
#     case = cfg.case
#     fname = f"Case_{case}_K_{arms}_k_{k}_d_{d}_T_{T}_explored_{cfg.init_explore}_noise_{sigma}"

#     # regret_results = dict()
#     # for agent_type in AGENTS:
#     #     regrets = run_trials(agent_type=agent_type,
#     #                          trials=cfg.trials,
#     #                          horizon=T,
#     #                          k=k,
#     #                          d=d,
#     #                          arms=arms,
#     #                          noise_std=cfg.reward_std,
#     #                          random_state=SEED,
#     #                          verbose=True)
#     #     key = AGENT_DICT[agent_type]
#     #     regret_results[key] = regrets

#     # # Function to run trials for a single agent
#     # def run_agent(agent_type):
#     #     regrets = run_trials(
#     #         agent_type=agent_type,
#     #         trials=cfg.trials,
#     #         horizon=T,
#     #         k=k,
#     #         d=d,
#     #         arms=arms,
#     #         noise_std=cfg.reward_std,
#     #         case=case,
#     #         random_state=SEED,
#     #         verbose=True,
#     #         fname=fname
#     #     )
#     #     key = AGENT_DICT[agent_type]
#     #     return key, regrets

#     # Parallel execution using ProcessPoolExecutor
#     regret_results = dict()
#     with ProcessPoolExecutor(max_workers=8) as executor:
#         results = executor.map(run_agent, AGENTS)

#     # Collect results
#     for key, regrets in results:
#         regret_results[key] = regrets

#     fig = show_result(regrets=regret_results, horizon=T, fontsize=15)

#     save_plot(fig, path=FIGURE_PATH, fname=fname)
