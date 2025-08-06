from cfg import get_cfg
from LowOFULmodels import *
from util import *

MOTHER_PATH = "."

DIST_DICT = {"gaussian": "g", "uniform": "u"}

AGENT_DICT = {
    "mab_ucb": r"UCB($\delta$)",
    "linucb": "LinUCB",
    "lints": "LinTS",
    "rolf_lasso": "RoLF-Lasso (Kim & Park)",
    "rolf_ridge": "RoLF-Ridge (Kim & Park)",
    "birolf_lasso": "BiRoLF-Lasso (Ours)",
    "dr_lasso": "DRLasso",
}

cfg = get_cfg()

RESULT_PATH = f"{MOTHER_PATH}/results/{str(cfg.date)}/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"
FIGURE_PATH = f"{MOTHER_PATH}/figures/{str(cfg.date)}/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"
LOG_PATH = (
    f"{MOTHER_PATH}/logs/{str(cfg.date)}/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"
)

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
    random_state: int,
):
    ## sample the true, observable, and unobservable features
    d_u = d_x_star - d_x  # dimension of unobserved feature(x)
    d_v = d_y_star - d_y  # dimension of unobserved feature(y)

    assert case in [1, 2, 3, 4, 5, 6, 7, 8, 9], "There exists only Case 1 to 9."

    ## For feature x
    if case in [1, 2, 3]:
        ## X is Default case
        np.random.seed(random_state)
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
        np.random.seed(random_state + 17)
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
        np.random.seed(random_state + 31)
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
        np.random.seed(random_state)
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
        np.random.seed(random_state + 17)

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
        np.random.seed(random_state + 31)
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


def bilinear_run_trials(
    agent_type: str,
    trials: int,
    horizon: int,
    d_x_star: int,
    d_x: int,
    M: int,
    d_y_star: int,
    d_y: int,
    N: int,
    noise_std: float,
    case: int,
    random_state: int,
    verbose: bool,
    fname: str,
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
    regret_container = np.zeros(trials, dtype=object)
    for trial in range(trials):
        ### Setting random state (Manual Folded)
        if random_state is not None:
            random_state_ = random_state + (513 * trial)
        else:
            random_state_ = None

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
                    random_state=random_state_,
                    explore=cfg.explore,
                    init_explore=exp_map[cfg.init_explore],
                )
            else:
                agent = RoLFLasso(
                    d=total_obs_dim,
                    arms=total_arms,
                    p=cfg.p,
                    delta=cfg.delta,
                    sigma=noise_std,
                    random_state=random_state_,
                )

        elif agent_type == "rolf_ridge":
            if cfg.explore:
                agent = RoLFRidge(
                    d=total_obs_dim,
                    arms=total_arms,
                    p=cfg.p,
                    delta=cfg.delta,
                    sigma=noise_std,
                    random_state=random_state_,
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
                    random_state=random_state_,
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
                    random_state=random_state_,
                    delta=cfg.delta,
                    p=cfg.p,
                    explore=cfg.explore,
                    init_explore=exp_map[cfg.init_explore],
                    theoretical_init_explore=False,
                )
            else:
                agent = BiRoLFLasso(
                    M=M,
                    N=N,
                    sigma=noise_std,
                    random_state=random_state_,
                    delta=cfg.delta,
                    p=cfg.p,
                    theoretical_init_explore=False,
                )

        elif agent_type == "low_oful":

            ## TODO: adjust the hyperparameter
            agent = ESTRLowOFUL(
                d1=d_x,
                d2=d_y,
                r=(int)(0.75 * min(d_x, d_y)),
                T1=(int)(horizon * 1 / 3),
                lam=0.8,
                lam_perp=0.8,
                B=1.0,
                B_perp=1.0,
                delta=0.99,
                sigma=1.0,
                random_state=123,
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
            random_state=random_state_ + 1,
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
            random_state=random_state_,
        )

        ## (M,N) matrix with the maximum absolute value does not exceed 1
        exp_rewards_mat = X_star.T @ reward_param_mat @ Y_star
        exp_rewards_mat = exp_rewards_mat / np.max(np.abs(exp_rewards_mat))

        if (
            isinstance(agent, LinUCB)
            or isinstance(agent, LinTS)
            or isinstance(agent, DRLassoBandit)
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

        regrets = bilinear_run(
            trial=trial,
            agent=agent,
            horizon=horizon,
            exp_rewards_mat=exp_rewards_mat,
            x=data_x,
            y=data_y,
            noise_dist=cfg.reward_dist,
            noise_std=noise_std,
            random_state=random_state_,
            verbose=verbose,
            fname=fname,
        )

        regret_container[trial] = regrets
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
    random_state: int,
    verbose: bool,
    fname: str,
):
    # x, y: augmented feature if the agent is RoLF (M, M), (N, N) each.
    regrets = np.zeros(horizon, dtype=float)

    if not verbose:
        bar = tqdm(range(horizon))
    else:
        bar = range(horizon)

    # For linear contextual bandits
    # For RoLF this is (MN,MN), otherwise (MN,d_x*d_y)
    z = np.kron(x, y)

    for t in bar:
        if random_state is not None:
            random_state_ = random_state + int(113 * t)
        else:
            random_state_ = None

        # if t == 0:
        #     print(f"Number of actions : {x.shape[0]}\tReward range : [{np.amin(exp_rewards):.5f}, {np.amax(exp_rewards):.5f}]")

        ## compute the optimal action
        optimal_action = np.argmax(exp_rewards_mat)
        M, N = exp_rewards_mat.shape
        optimal_i, optimal_j = action_to_ij(optimal_action, N)
        optimal_reward = exp_rewards_mat[optimal_i][optimal_j]

        ## choose the best action
        noise = subgaussian_noise(
            distribution=noise_dist, size=1, std=noise_std, random_state=random_state_
        )

        if isinstance(agent, (BiRoLFLasso, ESTRLowOFUL)):
            chosen_action = agent.choose(x, y)
        elif isinstance(agent, ContextualBandit):
            chosen_action = agent.choose(z)
        else:
            chosen_action = agent.choose()
        chosen_i, chosen_j = action_to_ij(chosen_action, N)
        chosen_reward = exp_rewards_mat[chosen_i][chosen_j] + noise

        # HERE
        if t % 10 == 0 and verbose:
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

        ## update the agent
        if isinstance(agent, (BiRoLFLasso, ESTRLowOFUL)):
            agent.update(x=x, y=y, r=chosen_reward)
        elif isinstance(agent, ContextualBandit):
            agent.update(x=z, r=chosen_reward)
        else:
            agent.update(a=chosen_action, r=chosen_reward)

    return np.cumsum(regrets)


## nothing change compare to show_result()
def bilinear_show_result(
    regrets: dict, horizon: int, figsize: tuple = (6, 5), fontsize=11
):

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["blue", "orange", "green", "red", "purple", "black"]
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
def bilinear_run_agent(agent_type):
    regrets = bilinear_run_trials(
        agent_type=agent_type,
        trials=cfg.trials,
        horizon=cfg.horizon,
        d_x_star=cfg.true_dim_x,
        d_x=cfg.dim_x,
        M=cfg.arm_x,
        d_y_star=cfg.true_dim_y,
        d_y=cfg.dim_y,
        N=cfg.arm_y,
        noise_std=cfg.reward_std,
        case=cfg.case,
        random_state=cfg.seed,
        verbose=True,
        fname=f"Case_{cfg.case}_Agent_{agent_type}_M_{cfg.arm_x}_N_{cfg.arm_y}_xstar_{cfg.true_dim_x}_ystar_{cfg.true_dim_y}_dx_{cfg.dim_x}_dy_{cfg.dim_y}_T_{cfg.horizon}_explored_{cfg.init_explore}_noise_{cfg.reward_std}",
    )
    key = AGENT_DICT[agent_type]
    return key, regrets


if __name__ == "__main__":
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
        "birolf_lasso",
        "rolf_lasso",
        "rolf_ridge",
        "dr_lasso",
        "linucb",
        "lints",
        "mab_ucb",
        "low_oful",
    ]
    case = cfg.case

    fname = f"Case_{case}_M_{M}_N_{N}_xstar_{d_x_star}_ystar_{d_y_star}_dx_{d_x}_dy_{d_y}_T_{T}_explored_{cfg.init_explore}_noise_{sigma}"

    # regret_results = dict()
    # for agent_type in AGENTS:
    #     regrets = run_trials(agent_type=agent_type,
    #                          trials=cfg.trials,
    #                          horizon=T,
    #                          k=k,
    #                          d=d,
    #                          arms=arms,
    #                          noise_std=cfg.reward_std,
    #                          random_state=SEED,
    #                          verbose=True)
    #     key = AGENT_DICT[agent_type]
    #     regret_results[key] = regrets

    # # Function to run trials for a single agent
    # def run_agent(agent_type):
    #     regrets = run_trials(
    #         agent_type=agent_type,
    #         trials=cfg.trials,
    #         horizon=T,
    #         k=k,
    #         d=d,
    #         arms=arms,
    #         noise_std=cfg.reward_std,
    #         case=case,
    #         random_state=SEED,
    #         verbose=True,
    #         fname=fname
    #     )
    #     key = AGENT_DICT[agent_type]
    #     return key, regrets

    # Parallel execution using ProcessPoolExecutor
    regret_results = dict()
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(bilinear_run_agent, AGENTS)

    # Collect results
    for key, regrets in results:
        regret_results[key] = regrets

    fig = bilinear_show_result(regrets=regret_results, horizon=T, fontsize=15)

    save_plot(fig, path=FIGURE_PATH, fname=fname)
    save_result(
        result=(vars(cfg), regret_results),
        path=RESULT_PATH,
        fname=fname,
        filetype=cfg.filetype,
    )

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
#                 )
#             else:
#                 agent = RoLFLasso(
#                     d=d,
#                     arms=arms,
#                     p=cfg.p,
#                     delta=cfg.delta,
#                     sigma=noise_std,
#                     random_state=random_state_,
#                 )

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
#     save_result(
#         result=(vars(cfg), regret_results),
#         path=RESULT_PATH,
#         fname=fname,
#         filetype=cfg.filetype,
#     )
