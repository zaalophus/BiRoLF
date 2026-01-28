import numpy as np
from util import *
from abc import ABC, abstractmethod
try:
    from calculate_alpha import *
except Exception:
    def linucb_alpha(delta: float):
        return 1.0
    def lints_alpha(d: int, reward_std: float, delta: float):
        return 1.0

import scipy
from sklearn.linear_model import Lasso, LinearRegression
import statsmodels.api as sm
from typing import Callable, Dict
import time
from collections import defaultdict

# Changelog (ICML26 alignment):
# - Switched BiRoLF coupling/importance weights to single p and matched-only imputation updates.
# - Blockwise main update keeps mu=lambda_t/Gamma_t and kappa_max from max abs entry.
# - Batched ou/uo block solvers reduce Python overhead while preserving the same objective.

# Global timing tracking variables - will be set by main.py
# TIMING_DATA = None
# shared_timing_data = None

#############################################################################
############################ Multi-Armed Bandits ############################
#############################################################################
class MAB(ABC):
    @abstractmethod
    def choose(self):
        pass

    @abstractmethod
    def update(self, a, r):
        pass


class eGreedyMAB(MAB):
    def __init__(
        self, arms: int, epsilon: float, alpha: float = 1.0, initial: float = 0
    ):
        self.arms = arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.initial = initial
        self.counts = np.zeros(self.arms)
        self.values = np.zeros(self.arms) + self.initial
        self.t = 0

    def choose(self):
        self.t += 1
        # print(f"Round : {self.t}, Epsilon: {self.epsilon}")
        if np.random.random() < self.epsilon:
            return np.random.choice(self.arms)
        else:
            (argmaxes,) = np.where(self.values == np.max(self.values))
            return np.random.choice(argmaxes)

    def update(self, a, r):
        ## count update
        self.counts[a] += 1

        ## value update
        value = self.values[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.values[a] = new_value

        ## epsilon update
        self.epsilon *= self.alpha


class ETC(MAB):
    ## Explore-then-Commit
    def __init__(self, arms: int, explore: int, horizon: int, initial: float = 0):
        assert (
            explore * arms <= horizon
        ), "Explore must be less than or equal to horizon"
        self.explore = explore
        self.arms = arms
        self.initial = initial
        self.counts = np.zeros(self.arms)
        self.values = np.zeros(self.arms) + self.initial
        self.t = 0

    def choose(self):
        ## Exploration Step
        self.t += 1
        if (self.t - 1) <= self.explore * self.arms:
            return (self.t - 1) % self.arms

        ## Exploitation Step
        (argmaxes,) = np.where(self.values == np.max(self.values))
        return np.random.choice(argmaxes)

    def update(self, a, r):
        ## count update
        self.counts[a] += 1

        ## value update
        value = self.values[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.values[a] = new_value


class UCBNaive(MAB):
    def __init__(
        self, n_arms: int, sigma: float = 0.1, alpha: float = 0.1, delta: float = 0.1
    ):
        self.n_arms = n_arms
        self.alpha = alpha
        self.delta = delta
        self.sigma = sigma
        self.counts = np.zeros(self.n_arms)
        self.qs = np.zeros(self.n_arms)
        self.ucbs = np.array([np.iinfo(np.int32).max for _ in range(self.n_arms)])
        self.t = 0

    def choose(self):
        self.t += 1
        returns = self.qs + self.ucbs
        argmaxes = np.where(returns == np.max(returns))[0]
        return np.random.choice(argmaxes)

    def update(self, a: int, r: float):
        """
        a: index of the chosen arm
        r: reward of the chosen arm
        """
        ## count update
        self.counts[a] += 1

        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.qs[a] = new_value

        ## ucb update
        inside = 2 * (self.sigma**2) * np.log(self.t / self.delta)
        self.ucbs[a] = self.alpha * np.sqrt(inside)


class UCBDelta(UCBNaive):
    def __init__(self, n_arms: int, delta: float):
        # set default values for sigma and alpha
        self.n_arms = n_arms
        self.delta = delta
        super().__init__(self.n_arms, delta=self.delta)

    def update(self, a: int, r: float):
        """
        a: index of the chosen arm
        r: reward of the chosen arm
        """
        ## count update
        self.counts[a] += 1

        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.qs[a] = new_value

        ## ucb update
        numerator = 2 * np.log(1 / self.delta)
        self.ucbs[a] = np.sqrt(numerator / self.counts[a])


class UCBAsymptotic(UCBNaive):
    def __init__(self, arms: int):
        self.arms = arms
        super().__init__(self.n_arms)

    def update(self, a, r):
        ## count update
        self.counts[a] += 1

        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.qs[a] = new_value

        ## ucb update
        ft = 1 + (self.t * (np.log(self.t) ** 2))
        numerator = 2 * np.log(ft)
        self.ucbs[a] = np.sqrt(numerator / self.counts[a])


class UCBMOSS(UCBNaive):
    def __init__(self, arms: int, horizon: int):
        self.arms = arms
        self.horizon = horizon
        super().__init__(self.n_arms)

    def update(self, a, r):
        ## count update
        self.counts[a] += 1

        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.qs[a] = new_value

        ## ucb update
        left = 4 / n
        right = np.log(np.maximum(1, (self.horizon / (self.n_arms * n))))
        self.ucbs[a] = np.sqrt(left * right)


class ThompsonSampling(MAB):
    def __init__(self, arms: int, distribution: str):
        self.arms = arms
        assert distribution.lower() in [
            "bernoulli",
            "gaussian",
        ], "Distribution must be either Bernoulli or Gaussian"
        self.distribution = distribution
        if distribution.lower() == "bernoulli":
            self.alphas = np.ones(self.arms)
            self.betas = np.ones(self.arms)
        elif distribution.lower() == "gaussian":
            self.mus = np.zeros(self.arms)
            self.sigmas = np.ones(self.arms)
        self.counts = np.zeros(shape=self.arms)
        self.qs = np.zeros(shape=self.arms)

    def choose(self):
        if self.distribution.lower() == "bernoulli":
            thetas = np.array(
                [
                    np.random.beta(a=alpha, b=beta)
                    for (alpha, beta) in zip(self.alphas, self.betas)
                ]
            )
        elif self.distribution.lower() == "gaussian":
            thetas = np.array(
                [
                    np.random.normal(loc=mu, scale=var)
                    for (mu, var) in zip(self.mus, self.sigmas)
                ]
            )
        (argmaxes,) = np.where(thetas == np.max(thetas))
        return np.random.choice(argmaxes)

    def update(self, a, r):
        ## count update
        self.counts[a] += 1

        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n - 1) / n) * value) + ((1 / n) * r)
        self.qs[a] = new_value

        ## parameter update
        if self.distribution.lower() == "bernoulli":
            self.alphas[a] += r
            self.betas[a] += 1 - r
        elif self.distribution.lower() == "gaussian":
            self.mus[a] = new_value
            self.sigmas[a] = np.sqrt(1 / n)


#############################################################################
############################ Contextual Bandits #############################
#############################################################################
class ContextualBandit(ABC):
    @abstractmethod
    def choose(self, x):
        pass

    @abstractmethod
    def update(self, x, r):
        pass


class LinUCB(ContextualBandit):
    def __init__(self, d: int, lbda: float, delta: float) -> None:
        self.d = d
        self.xty = np.zeros(d)
        self.Vinv = (1 / lbda) * np.identity(d)
        self.theta_hat = np.zeros(d)
        self.delta = delta
        self.t = 0
        self._alpha_base = linucb_alpha(delta=self.delta)

    def choose(self, x: np.ndarray) -> int:
        # x: action set at each round (N, d)
        self.t += 1

        ## compute the ridge estimator
        self.theta_hat = self.Vinv @ self.xty

        ## compute the ucb scores for each arm
        alpha = self._alpha_base * np.sqrt(np.log(self.t))
        expected = x @ self.theta_hat  # (N, ) theta_T @ x_t
        xV = x @ self.Vinv
        width = np.sqrt(np.sum(xV * x, axis=1))  # (N, ) widths
        ucb_scores = expected + (alpha * width)  # (N, ) ucb score

        ## chose the argmax the ucb score
        maximum = np.max(ucb_scores)
        argmax = np.flatnonzero(ucb_scores == maximum)
        self.chosen_action = np.random.choice(argmax)
        return self.chosen_action

    def update(self, x: np.ndarray, r: float) -> None:
        # x: context of the chosen action (d, )
        chosen_context = x[self.chosen_action, :]
        self.Vinv = shermanMorrison(self.Vinv, chosen_context)
        self.xty += r * chosen_context

    def __get_param(self):
        return {"param": self.theta_hat}


class LinTS(ContextualBandit):
    def __init__(
        self, d: int, lbda: float, horizon: int, reward_std: float, delta: float
    ) -> None:
        self.d = d
        self.Binv = (1 / lbda) * np.identity(d)
        self.xty = np.zeros(d)
        self.theta_hat = np.zeros(d)
        self.horizon = horizon
        self.reward_std = reward_std
        self.delta = delta
        self.t = 0
        self._alpha_base = lints_alpha(d=self.d, reward_std=self.reward_std, delta=self.delta)

    def choose(self, x: np.ndarray) -> int:
        # x: action set at each round (N, d)
        self.t += 1

        ## compute the ridge estimator
        self.theta_hat = self.Binv @ self.xty

        ## parameter sampling
        # self.alpha_ = self.alpha * np.sqrt(np.log(self.t))
        # alpha = lints_alpha(d=self.d, horizon=self.horizon, reward_std=self.reward_std, delta=self.delta) * np.sqrt(np.log(self.t))
        alpha = self._alpha_base
        try:
            L = np.linalg.cholesky(self.Binv)
            z = np.random.randn(self.d)
            tilde_theta = self.theta_hat + alpha * (L @ z)
        except np.linalg.LinAlgError:
            tilde_theta = np.random.multivariate_normal(
                mean=self.theta_hat, cov=(alpha**2) * self.Binv
            )

        ## compute estimates and choose the argmax
        expected = x @ tilde_theta  # (N, ) vector
        maximum = np.max(expected)
        argmax = np.flatnonzero(expected == maximum)
        self.chosen_action = np.random.choice(argmax)
        return self.chosen_action

    def update(self, x: np.ndarray, r: float) -> None:
        # x: (K, d)
        # r: reward seen (scalar)
        chosen_context = x[self.chosen_action, :]
        self.Binv = shermanMorrison(self.Binv, chosen_context)
        self.xty += r * chosen_context

    def __get_param(self):
        return {"param": self.theta_hat}


class RoLFLasso(ContextualBandit):
    def __init__(
        self,
        d: int,
        arms: int,
        p: float,
        delta: float,
        sigma: float,
        explore: bool = False,
        init_explore: int = 0,
        lam_c_impute: float = 1.0,
        lam_c_main: float = 1.0,
    ):
        self.lam_c_impute = lam_c_impute
        self.lam_c_main = lam_c_main

        self.t = 0
        self.d = d
        self.K = arms
        self.mu_hat = np.zeros(self.K)  # main estimator
        self.mu_check = np.zeros(self.K)  # imputation estimator
        self.impute_prev = np.zeros(self.K)
        self.main_prev = np.zeros(self.K)
        self.sigma = sigma  # variance of noise
        self.p = p  # hyperparameter for action sampling
        self.delta = delta  # confidence parameter
        self.action_history = []  # history of chosen actions up to the current round
        self.reward_history = []  # history of observed rewards up to the current round
        self.matching = (
            dict()
        )  # history of rounds that the pseudo action and the chosen action matched
        self.explore = explore
        self.init_explore = init_explore
        self.fista_max_iter = 200
        self.fista_tol = 1e-6
        self._arm_indices = np.arange(self.K)
        self._static_arms_initialized = False
        self._profile_ops = False
        self.X_static = None
        self.G_main_base = None
        self.lam_main_max = None
        self.Gamma_main = 0
        self.sum_xx = None
        self.sum_r_x = None
        self.b_main = None
        self.G_imp = None
        self.b_imp = None

    def _init_main_cache(self, x: np.ndarray) -> None:
        if self._static_arms_initialized:
            return
        self.X_static = x
        self.d = int(self.X_static.shape[1])
        self.G_main_base = self.X_static.T @ self.X_static
        self.lam_main_max = float(np.max(np.linalg.eigvalsh(self.G_main_base))) if self.G_main_base.size else 0.0
        self.sum_xx = np.zeros((self.d, self.d), dtype=float)
        self.sum_r_x = np.zeros(self.d, dtype=float)
        self.b_main = np.zeros(self.d, dtype=float)
        self.G_imp = np.zeros((self.d, self.d), dtype=float)
        self.b_imp = np.zeros(self.d, dtype=float)
        self._static_arms_initialized = True

    def choose(self, x: np.ndarray):
        # x : (K, d) augmented feature matrix where each row denotes the augmented features
        self.t += 1

        ## compute the \hat{a}_t
        if self.explore:
            if self.t > self.init_explore:
                decision_rule = x @ self.mu_hat
                # print(f"Decision rule : {decision_rule}")
                a_hat = np.argmax(decision_rule)
            else:
                a_hat = np.random.choice(np.arange(self.K))
        else:
            decision_rule = x @ self.mu_hat
            # print(f"Decision rule : {decision_rule}")
            a_hat = np.argmax(decision_rule)

        self.a_hat = a_hat
        self._ahat_history = getattr(self, "_ahat_history", {})
        self._ahat_history[self.t] = a_hat

        ## sampling actions (resample chosen and pseudo until match or max_iter)
        denom = np.log(1.0 / max(1.0 - self.p, 1e-12))
        max_iter = int(
            np.log(2.0 * ((self.t + 1) ** 2) / self.delta) / max(denom, 1e-12)
        )

        chosen_dist = np.full(self.K, (1.0 / np.sqrt(self.t)) / max(self.K - 1, 1), dtype=float)
        chosen_dist[a_hat] = 1 - (1.0 / np.sqrt(self.t))

        pseudo_action = -1
        chosen_action = -2
        count = 0
        while (pseudo_action != chosen_action) and (count <= max_iter):
            chosen_action = np.random.choice(self._arm_indices, p=chosen_dist).item()
            pseudo_dist = np.full(self.K, (1.0 - self.p) / max(self.K - 1, 1), dtype=float)
            pseudo_dist[chosen_action] = self.p
            pseudo_action = np.random.choice(self._arm_indices, p=pseudo_dist).item()
            count += 1
        self.pseudo_action = pseudo_action
        self.chosen_action = chosen_action
        return chosen_action

    def update(self, x: np.ndarray, r: float):
        # x : (K, K) augmented feature matrix
        # r : reward of the chosen_action
        self._last_impute_time = 0.0
        self._last_main_time = 0.0
        self._last_impute_iters = 0
        self._last_main_iters = 0
        if self.pseudo_action != self.chosen_action:
            self.matching[self.t] = (
                (self.pseudo_action == self.chosen_action),
                None,
                None,
                None,
                None,
            )
            return

        self.action_history.append(self.chosen_action)
        self.reward_history.append(r)
        self._init_main_cache(x)

        # lam_impute = 2 * self.p * self.sigma * np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta))
        # lam_main = (1 + 2 / self.p) * self.sigma * np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta))

        # lam_impute = self.p * np.sqrt(np.log(self.t))
        # lam_main = self.p * np.sqrt(np.log(self.t))

        lam_impute = self.lam_c_impute * (2 * self.p * self.sigma *
              np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta)))
        lam_main   = self.lam_c_main * ((1 + 2 / self.p) * self.sigma *
              np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta)))

        ## compute the imputation estimator (FISTA on quadratic form)
        chosen_context = x[self.chosen_action, :]
        self.G_imp += np.outer(chosen_context, chosen_context)
        self.b_imp += r * chosen_context
        L_imp = 2.0 * max(float(np.max(np.linalg.eigvalsh(self.G_imp))), 1e-12)
        impute_stats = {} if getattr(self, "_profile_ops", False) else None
        impute_start = time.perf_counter()
        mu_impute = fista_lasso_vector(
            self.G_imp,
            self.b_imp,
            lam_impute,
            self.impute_prev,
            L_imp,
            self.fista_max_iter,
            self.fista_tol,
            use_fista=True,
            stats=impute_stats,
        )
        self._last_impute_time = time.perf_counter() - impute_start
        if impute_stats is not None:
            self._last_impute_iters = int(impute_stats.get("iters", 0))

        ## compute the pseudo rewards for the current data
        # Conditional pseudo sampling -> constant 1/p correction for unbiasedness.
        pseudo_rewards = x @ mu_impute
        pseudo_rewards[self.chosen_action] += (1 / self.p) * (
            r - (x[self.chosen_action, :] @ mu_impute)
        )
        self.matching[self.t] = (
            (self.pseudo_action == self.chosen_action),
            None,
            None,
            self.chosen_action,
            r,
        )

        ## compute the main estimator (FISTA on quadratic form)
        self.Gamma_main += 1
        chosen_context = x[self.chosen_action, :]
        self.sum_xx += np.outer(chosen_context, chosen_context)
        self.sum_r_x += r * chosen_context
        # Use current imputation estimate for all matched rounds (paper definition).
        self.b_main = (
            self.Gamma_main * (self.G_main_base @ mu_impute)
            + (1.0 / self.p) * self.sum_r_x
            - (1.0 / self.p) * (self.sum_xx @ mu_impute)
        )
        G_main = self.G_main_base * float(self.Gamma_main)
        L_main = 2.0 * max(self.lam_main_max, 1e-12) * float(self.Gamma_main)
        main_stats = {} if getattr(self, "_profile_ops", False) else None
        optimization_start_time = time.perf_counter()
        mu_main = fista_lasso_vector(
            G_main,
            self.b_main,
            lam_main,
            self.main_prev,
            L_main,
            self.fista_max_iter,
            self.fista_tol,
            use_fista=True,
            stats=main_stats,
        )
        optimization_time = time.perf_counter() - optimization_start_time
        self._last_main_time = optimization_time
        if main_stats is not None:
            self._last_main_iters = int(main_stats.get("iters", 0))

        if not getattr(self, "_benchmark_mode", False) and hasattr(self, "_timing_data") and self._timing_data is not None:
            agent = self.__class__.__name__
            trial = getattr(self, "_trial", 0)
            timing_store = self._timing_data.get("optimization", self._timing_data)
            timing_store.setdefault(agent, {}).setdefault(trial, []).append(optimization_time)

        ## update the mu_hat
        self.mu_hat = mu_main
        self.mu_check = mu_impute
        self.impute_prev = mu_impute
        self.main_prev = mu_main

    def __imputation_loss(
        self, beta: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float
    ):
        residuals = (y - (X @ beta)) ** 2
        loss = np.sum(residuals, axis=0)
        l1_norm = vector_norm(beta, type="l1")
        return loss + (lam * l1_norm)

    # def __main_loss(self, beta:np.ndarray, lam:float, matching_history:dict):
    #     ## matching_history : dict[t] = (bool, X, y) - bool denotes whether the matching event occurred or not
    #     loss = 0
    #     for key in matching_history:
    #         matched, X, pseudo_rewards, _, _ = matching_history[key]
    #         if matched:
    #             residuals = (pseudo_rewards - (X @ beta)) ** 2
    #             interim_loss = np.sum(residuals, axis=0)
    #         else:
    #             interim_loss = 0
    #         loss += interim_loss
    #     l1_norm = vector_norm(beta, type="l1")
    #     return loss + (lam * l1_norm)

    def __main_loss(self, beta: np.ndarray, lam: float, matching_history: dict):
        # Extract matched keys and data
        matched_keys = [
            key for key, value in matching_history.items() if value[0]
        ]  # Filter matched entries
        X_list = [
            matching_history[key][1] for key in matched_keys
        ]  # List of X matrices
        pseudo_rewards_list = [
            matching_history[key][2] for key in matched_keys
        ]  # List of pseudo_rewards

        # Compute residuals for matched keys
        residuals_list = [
            (pseudo_rewards - X @ beta) ** 2
            for X, pseudo_rewards in zip(X_list, pseudo_rewards_list)
        ]

        # Sum all residuals efficiently
        residuals_sum = sum(np.sum(residuals, axis=0) for residuals in residuals_list)

        # L1 regularization
        l1_norm = np.sum(np.abs(beta))

        # Total loss
        return residuals_sum + lam * l1_norm

    def __get_param(self):
        return {"param": self.mu_hat, "impute": self.mu_check}


class RoLFRidge(ContextualBandit):
    def __init__(
        self,
        d: int,
        arms: int,
        p: float,
        delta: float,
        sigma: float,
        explore: bool = False,
        init_explore: int = 0,
    ):
        self.t = 0
        self.d = d
        self.K = arms
        self.mu_hat = np.zeros(self.K)  # main estimator
        self.mu_check = np.zeros(self.K)  # imputation estimator
        self.sigma = sigma  # variance of noise
        self.p = p  # hyperparameter for action sampling
        self.delta = delta  # confidence parameter
        self.matching = (
            dict()
        )  # history of rounds that the pseudo action and the chosen action matched
        self.Vinv_impute = self.p * np.identity(self.K)
        self.xty_impute = np.zeros(self.K)
        self.explore = explore
        self.init_explore = init_explore
        self._static_arms_initialized = False
        self.X_static = None
        self.G_main = None
        self.G_eigvals = None
        self.G_eigvecs = None
        self.sum_pseudo = np.zeros(self.K)
        self.gamma = 0
        self._arm_indices = np.arange(self.K)
        self._profile_ops = False

    def _init_main_cache(self, x: np.ndarray) -> None:
        if self._static_arms_initialized:
            return
        # RoLF uses a fixed action feature matrix across rounds in this codebase.
        self.X_static = x
        self.G_main = self.X_static.T @ self.X_static
        self.G_eigvals, self.G_eigvecs = np.linalg.eigh(self.G_main)
        self._static_arms_initialized = True

    def choose(self, x: np.ndarray):
        # x : (K, d) augmented feature matrix where each row denotes the augmented features
        self.t += 1

        ## compute the \hat{a}_t
        if self.explore:
            if self.t > self.init_explore:
                decision_rule = x @ self.mu_hat
                # print(f"Decision rule : {decision_rule}")
                a_hat = np.argmax(decision_rule)
            else:
                a_hat = np.random.choice(np.arange(self.K))
        else:
            decision_rule = x @ self.mu_hat
            # print(f"Decision rule : {decision_rule}")
            a_hat = np.argmax(decision_rule)

        self.a_hat = a_hat
        self._ahat_history = getattr(self, "_ahat_history", {})
        self._ahat_history[self.t] = a_hat

        ## sampling actions (resample chosen and pseudo until match or max_iter)
        denom = np.log(1.0 / max(1.0 - self.p, 1e-12))
        max_iter = int(
            np.log(2.0 * ((self.t + 1) ** 2) / self.delta) / max(denom, 1e-12)
        )
        chosen_dist = np.array(
            [(1 / np.sqrt(self.t)) / (self.K - 1)] * self.K, dtype=float
        )
        chosen_dist[a_hat] = 1 - (1 / np.sqrt(self.t))

        pseudo_action = -1
        chosen_action = -2
        count = 0
        while (pseudo_action != chosen_action) and (count <= max_iter):
            chosen_action = np.random.choice(self._arm_indices, size=1, replace=False, p=chosen_dist).item()
            pseudo_dist = np.array([(1 - self.p) / (self.K - 1)] * self.K, dtype=float)
            pseudo_dist[chosen_action] = self.p
            pseudo_action = np.random.choice(self._arm_indices, size=1, replace=False, p=pseudo_dist).item()
            count += 1

        self.pseudo_action = pseudo_action
        self.chosen_action = chosen_action
        # print(f"Round: {self.t}, a_hat: {a_hat}, pseudo_action: {pseudo_action}, chosen_action: {chosen_action}, count: {count}")
        return chosen_action

    def update(self, x: np.ndarray, r: float):
        # x : (K, K) augmented feature matrix
        # r : reward of the chosen_action
        self._last_impute_time = 0.0
        self._last_main_time = 0.0
        self._last_impute_iters = 0
        self._last_main_iters = 0
        if self.pseudo_action == self.chosen_action:
            self._init_main_cache(x)
            ## compute the imputation estimator based on history
            impute_start = time.perf_counter()
            chosen_context = x[self.chosen_action, :]
            self.Vinv_impute = shermanMorrison(self.Vinv_impute, chosen_context)
            self.xty_impute += r * chosen_context
            mu_impute = self.Vinv_impute @ self.xty_impute
            self._last_impute_time = time.perf_counter() - impute_start

            ## compute the pseudo rewards for the current data
            pseudo_rewards = x @ mu_impute
            pseudo_rewards[self.chosen_action] += (1 / self.p) * (
                r - (x[self.chosen_action, :] @ mu_impute)
            )
            self.matching[self.t] = (
                (self.pseudo_action == self.chosen_action),
                None,
                None,
                self.chosen_action,
                r,
            )
            self.sum_pseudo += pseudo_rewards
            self.gamma += 1

            ## compute the main estimator using cached eigendecomposition
            main_start = time.perf_counter()
            b_main = self.X_static.T @ self.sum_pseudo
            denom = 1.0 + self.gamma * self.G_eigvals
            mu_main = self.G_eigvecs @ ((self.G_eigvecs.T @ b_main) / denom)
            self._last_main_time = time.perf_counter() - main_start

            ## update the mu_hat
            self.mu_hat = mu_main
            self.mu_check = mu_impute
        else:
            self.matching[self.t] = (
                (self.pseudo_action == self.chosen_action),
                None,
                None,
                None,
                None,
            )

    # def __main_estimation(self, matching_history:dict, dimension:int):
    #     ## matching_history : dict[t] = (bool, X, y) - bool denotes whether the matching event occurred or not
    #     inv = np.identity(dimension)
    #     score = np.zeros(dimension, dtype=float)
    #     for key in matching_history:
    #         matched, X, pseudo_rewards, _, _ = matching_history[key]
    #         if matched:
    #             # inverse matrix
    #             inv_init = np.zeros(shape=(dimension, dimension))
    #             for a in range(X.shape[0]):
    #                 inv_init += np.outer(X[a, :], X[a, :])
    #             inv += inv_init

    #             # score
    #             score_init = np.zeros(shape=dimension, dtype=float)
    #             for a in range(X.shape[0]):
    #                 score_init += pseudo_rewards[a] * X[a, :]
    #             score += score_init

    #     return scipy.linalg.inv(inv) @ score

    def __main_estimation(self, matching_history: dict, dimension: int):
        # Initialize inv and score
        inv = np.identity(dimension)
        score = np.zeros(dimension, dtype=float)

        # Filter matched entries
        matched_entries = [value for key, value in matching_history.items() if value[0]]

        # Process matched entries
        for _, X, pseudo_rewards, _, _ in matched_entries:
            # Update inv (outer products of rows in X)
            inv += X.T @ X

            # Update score (weighted sum of rows in X)
            score += X.T @ pseudo_rewards

        # Compute final estimation
        return scipy.linalg.inv(inv) @ score

    def __get_param(self):
        return {"param": self.mu_hat, "impute": self.mu_check}


class DRLassoBandit(ContextualBandit):
    def __init__(
        self, d: int, arms: int, lam1: float, lam2: float, zT: float, tr: bool
    ):
        ## learning params
        self.d = d
        self.arms = arms
        self.lam1 = lam1
        self.lam2 = lam2
        self.tr = tr
        self.zT = zT

        ## initialization
        self.beta_prev = np.zeros(self.d)
        self.beta_hat = np.zeros(self.d)
        self.pi_t = 0
        self.x = []  # containing context history
        self.r = []  # containing reward history
        self.t = 0  # learning round
        # FISTA-based lasso update (fast)
        self.fista_max_iter = 200
        self.fista_tol = 1e-6
        self.G = np.zeros((self.d, self.d), dtype=float)
        self.b = np.zeros(self.d, dtype=float)
        self.G_trace = 0.0
        self._last_bar_x = None
        self._last_main_time = 0.0
        self._last_main_iters = 0
        self._profile_ops = False

    def choose(self, x):
        ## x : (K, d) array - all contexts observed at t
        self.t += 1
        if self.t <= self.zT:
            # forced sampling
            self.action = np.random.choice(self.arms, replace=False)
            self.pi_t = 1 / self.arms
        else:
            # UCB
            expected_reward = x @ self.beta_hat  # (K, ) array
            lam1 = self.lam1 * np.sqrt((np.log(self.t) + np.log(self.d)) / self.t)
            lam1 = np.minimum(1, np.maximum(0, lam1))
            self.mt = np.random.choice([0, 1], p=[1 - lam1, lam1])
            if self.mt == 1:
                self.action = np.random.choice(self.arms)
            else:
                self.action = np.argmax(expected_reward)

            self.pi_t = (lam1 / self.arms) + (
                (1 - lam1) * (self.action == np.argmax(expected_reward))
            )

        bar_x = np.mean(x, axis=0)
        self.x.append(bar_x)
        self._last_bar_x = bar_x
        self.rhat = x @ self.beta_hat

        return self.action

    def update(self, x, r):
        ## x : (K, d) array - context of the all actions in round t
        ## r : float - reward
        self._last_main_time = 0.0
        self._last_main_iters = 0
        r_hat = np.mean(self.rhat) + (
            (r - (x[self.action] @ self.beta_hat)) / (self.arms * self.pi_t)
        )
        if self.tr:
            r_hat = np.minimum(3.0, np.maximum(-3.0, r_hat))
        self.r.append(r_hat)

        lam2 = self.lam2 * np.sqrt((np.log(self.t) + np.log(self.d)) / self.t)
        # FISTA on quadratic form: min beta^T G beta - 2 b^T beta + lam ||beta||_1
        if self._last_bar_x is None:
            bar_x = np.mean(x, axis=0)
        else:
            bar_x = self._last_bar_x
        self.G += np.outer(bar_x, bar_x)
        self.G_trace += float(bar_x @ bar_x)
        self.b += r_hat * bar_x

        L = 2.0 * max(self.G_trace, 1e-12)  # safe Lipschitz upper bound
        stats = {} if getattr(self, "_profile_ops", False) else None
        start = time.perf_counter()
        self.beta_hat = fista_lasso_vector(
            self.G,
            self.b,
            lam2,
            self.beta_prev,
            L,
            self.fista_max_iter,
            self.fista_tol,
            use_fista=True,
            stats=stats,
        )
        self._last_main_time = time.perf_counter() - start
        if stats is not None:
            self._last_main_iters = int(stats.get("iters", 0))
        self.beta_prev = self.beta_hat

    def __lasso_loss(self, beta: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float):
        loss = np.sum((y - X @ beta) ** 2, axis=0)
        l1norm = np.sum(np.abs(beta))
        return loss + (lam * l1norm)

    def __get_param(self):
        return self.beta_hat


class LassoBandit(ContextualBandit):
    def __init__(
        self,
        arms: int,
        horizon: int,
        d: int,
        q: int,
        h: float,
        lam1: float,
        lam2: float,
    ):
        ## input params for algorithms
        self.q = q  # input param 1 - for forced-sampling
        self.h = h  # input param 2
        self.lam1 = lam1  # input param 3
        self.lam2 = lam2  # input param 4

        ## basic params for bandits
        self.arms = arms  # the number of arms
        self.horizon = horizon  # learning horizon
        self.d = d  # dimension of features
        self.t = 0  # learning round; t <= horizon
        self.n = 0  # sample size

        ## sets
        self.Tx = {i: [] for i in range(self.arms)}
        self.Sx = {i: [] for i in range(self.arms)}
        self.Tr = {i: [] for i in range(self.arms)}
        self.Sr = {i: [] for i in range(self.arms)}

        ## estmators
        self.beta_t = np.zeros((self.arms, d))  # forced-sample estimators
        self.beta_s = np.zeros((self.arms, d))  # all samples estimators
        self.lasso_t = Lasso(alpha=lam1)

    def choose(self, x: np.ndarray):
        ## x: (d, ) array - context vector of time t
        self.t += 1

        flag = (((2**self.n) - 1) * self.arms * self.q) + 1
        if self.t == flag:
            self.set = np.arange(self.t, self.t + (self.q * self.arms))
            self.n += 1

        if self.t in self.set:
            ## if t is in T_i for any i
            ind = list(self.set).index(self.t)
            self.action = ind // self.q
            self.Tx[self.action].append(x)
        else:
            ## if indices is none
            expected_T = self.beta_t @ x
            max_expected_T = np.amax(expected_T)
            K_hat = np.argwhere(expected_T >= (max_expected_T - (self.h / 2))).flatten()

            expected_S = self.beta_s @ x
            filtered_expected = expected_S[K_hat]
            argmax = np.argmax(filtered_expected)
            self.action = K_hat[argmax]

        self.Sx[self.action].append(
            x
        )  # append the context of the actually chosen action
        return self.action

    def update(self, r: float):
        if self.t in self.set:
            self.Tr[self.action].append(r)
            ## update beta_t using Lasso
            data_t, target_t = np.vstack(self.Tx[self.action]), np.array(
                self.Tr[self.action]
            )
            # print(data_t.shape)
            beta_t = scipy.optimize.minimize(
                self.__lasso_loss,
                np.zeros(d),
                args=(data_t, target_t, self.lam1),
                method="SLSQP",
                options={"disp": False, "ftol": 1e-6, "maxiter": 10000},
            ).x

            self.beta_t[self.action] = beta_t

        self.Sr[self.action].append(r)
        ## update beta_s using Lasso
        lam2_t = self.lam2 * np.sqrt(((np.log(self.t) + np.log(self.d)) / self.t))
        data_s, target_s = np.vstack(self.Sx[self.action]), np.array(
            self.Sr[self.action]
        )
        # print(f"action : {self.action}, data : {data_s.shape}")
        beta_s = scipy.optimize.minimize(
            self.__lasso_loss,
            np.zeros(d),
            args=(data_s, target_s, lam2_t),
            method="SLSQP",
            options={"disp": False, "ftol": 1e-6, "maxiter": 10000},
        ).x

        self.beta_s[self.action] = beta_s

    def __lasso_loss(self, beta: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float):
        # print(X)
        # print(f"X : {X.shape}, beta : {beta.shape}")
        loss = np.sum((y - X @ beta) ** 2, axis=0)
        l1norm = np.sum(np.abs(beta))
        return loss + (lam * l1norm)


class BiRoLFLasso_old(ContextualBandit):
    def __init__(
        self,
        M: int,
        N: int,
        sigma: float,
        delta: float,
        p: float,
        p1: float = None,
        p2: float = None,
        explore: bool = False,
        init_explore: int = 0,
        theoretical_init_explore: bool = False,
        lam_c_impute: float = 1.0,
        lam_c_main: float = 1.0,
    ):
        # Regularization scale multipliers (tunable from outside)
        self.lam_c_impute = lam_c_impute
        self.lam_c_main = lam_c_main

        self.t = 0
        self.explore = explore
        self.init_explore = init_explore
        ## TODO: make theoretical C_e
        if theoretical_init_explore:
            # self.init_explore = ((8*M*N)**3)
            pass
        self.M = M
        self.N = N
        self.delta = delta
        
        self.p = p
        # p1과 p2가 별도로 지정되지 않으면 p 값을 사용
        self.p1 = p1 if p1 is not None else p
        self.p2 = p2 if p2 is not None else p
        
        self.sigma = sigma

        self.action_i_history = []
        self.action_j_history = []
        self.reward_history = []

        self.matching = dict()
        self.Phi_hat = np.zeros((self.M, self.N))
        self.Phi_check = np.zeros((self.M, self.N))
        self.impute_prev = np.zeros((self.M, self.N))
        self.main_prev = np.zeros((self.M, self.N))
        self._arm_indices = np.arange(self.M * self.N)
        self._profile_ops = False

    def choose(self, x: np.ndarray, y: np.ndarray):
        # x : (M, M) augmented feature matrix where each row denotes the augmented features
        # y : (N, N) augmented feature matrix where each row denotes the augmented features

        self.t += 1

        ## compute the \hat{a}_t
        if self.explore:
            if self.t > self.init_explore:
                decision_rule = x @ self.Phi_hat @ y.T
                # print(f"Decision rule : {decision_rule}")
                a_hat = np.argmax(decision_rule)
            else:
                a_hat = np.random.choice(np.arange(self.M * self.N))
        else:
            ## decision_rule : (M,N)
            decision_rule = x @ self.Phi_hat @ y.T
            # print(f"Decision rule : {decision_rule}")
            a_hat = np.argmax(decision_rule)

        i_hat, j_hat = action_to_ij(a_hat, self.N)

        self.a_hat = a_hat
        self._ahat_history = getattr(self, "_ahat_history", {})
        self._ahat_history[self.t] = a_hat
        self.i_hat = i_hat
        self.j_hat = j_hat
        self._hat_history = getattr(self, "_hat_history", {})
        self._hat_history[self.t] = (self.i_hat, self.j_hat)

        ## sampling actions (resample chosen and pseudo until match or max_iter)
        total_arms = self.M * self.N
        denom = np.log(1.0 / max(1.0 - self.p, 1e-12))
        max_iter = int(
            np.log(2.0 * ((self.t + 1) ** 2) / self.delta) / max(denom, 1e-12)
        )

        chosen_dist = np.full(total_arms, (1.0 / np.sqrt(self.t)) / max(total_arms - 1, 1), dtype=float)
        chosen_dist[a_hat] = 1 - (1.0 / np.sqrt(self.t))

        pseudo_action = -1
        chosen_action = -2
        count = 0
        while (pseudo_action != chosen_action) and (count <= max_iter):
            chosen_action = np.random.choice(self._arm_indices, p=chosen_dist).item()
            pseudo_dist = np.full(total_arms, (1.0 - self.p) / max(total_arms - 1, 1), dtype=float)
            pseudo_dist[chosen_action] = self.p
            pseudo_action = np.random.choice(self._arm_indices, p=pseudo_dist).item()
            count += 1

        self.pseudo_action = pseudo_action
        self.chosen_action = chosen_action
        return chosen_action

    def update(self, x: np.ndarray, y: np.ndarray, r: float):
        # x : (M, M) augmented feature matrix
        # y : (N, N) augmented feature matrix
        # r : reward of the chosen_action
        self._last_impute_time = 0.0
        self._last_main_time = 0.0
        self._last_impute_iters = 0
        self._last_main_iters = 0
        if self.pseudo_action == self.chosen_action:
            chosen_i, chosen_j = action_to_ij(self.chosen_action, self.N)
            self.action_i_history.append(chosen_i)
            self.action_j_history.append(chosen_j)
            self.reward_history.append(r)

        # lam_impute = 2 * self.p * self.sigma * np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta))
        # lam_main = (1 + 2 / self.p) * self.sigma * np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta))

        # lam_impute = self.p * np.sqrt(np.log(self.t))
        # lam_main = self.p * np.sqrt(np.log(self.t))

        # lam_impute = self.p
        # lam_main = self.p

        kappa_x = np.max(np.abs(x))
        kappa_y = np.max(np.abs(y))

        lam_impute = self.lam_c_impute * (
            2 * self.sigma * kappa_x * kappa_y * np.sqrt(2 * self.t * np.log(2 * self.M * self.N * self.t**2 / self.delta))
        )
        lam_main = self.lam_c_main * (
            (4 * self.sigma * kappa_x * kappa_y / self.p) *
            np.sqrt(2 * self.t * np.log(2 * self.M * self.N * self.t**2 / self.delta))
        )

        if self.pseudo_action == self.chosen_action:
            ## compute the imputation estimator
            i_impute = x[self.action_i_history, :]  # (t, d_x) matrix
            j_impute = y[self.action_j_history, :]  # (t, d_y) matrix

            target_impute = np.array(self.reward_history)
            # print(f"gram_sqrt : {gram_sqrt.shape}")
            # print(f"impute_prev : {self.impute_prev.shape}")

            impute_shape = self.impute_prev.shape
            impute_start_time = time.perf_counter()
            Phi_impute = scipy.optimize.minimize(
                self.__imputation_loss,
                self.impute_prev.reshape(-1),
                args=(i_impute, j_impute, target_impute, lam_impute),
                method="SLSQP",
                options={"disp": False, "ftol": 1e-6, "maxiter": 10000},
            ).x.reshape(impute_shape)
            self._last_impute_time = time.perf_counter() - impute_start_time

            ## compute the pseudo rewards for the current data
            pseudo_rewards = x @ Phi_impute @ y.T
            chosen_i, chosen_j = action_to_ij(self.chosen_action, self.N)
            # Conditional pseudo sampling -> constant 1/p correction for unbiasedness.
            w_now = 1.0 / max(self.p, 1e-12)
            pseudo_rewards[chosen_i, chosen_j] += w_now * (
                r - (x[chosen_i, :] @ Phi_impute @ y[chosen_j, :].T)
            )
            self.matching[self.t] = (
                (self.pseudo_action == self.chosen_action),
                x,
                y,
                pseudo_rewards,
                self.chosen_action,
                r,
            )

            ## compute the main estimator
            
            # Time the lasso optimization for BiRoLFLasso_old
            optimization_start_time = time.perf_counter()
            
            main_prev_shape = self.main_prev.shape
            Phi_main = scipy.optimize.minimize(
                self.__main_loss,
                self.main_prev.reshape(-1),
                args=(lam_main, self.matching),
                method="SLSQP",
                options={"disp": False, "ftol": 1e-6, "maxiter": 10000},
            ).x.reshape(main_prev_shape)
            
            optimization_end_time = time.perf_counter()
            optimization_time = optimization_end_time - optimization_start_time
            self._last_main_time = optimization_time
            
            # Record timing data for ablation study
            if not getattr(self, "_benchmark_mode", False) and hasattr(self, '_timing_data') and self._timing_data is not None:
                agent_name = self.__class__.__name__
                trial = getattr(self, '_trial', 0)
                timing_store = self._timing_data.get("optimization", self._timing_data)

                # Initialize nested dict structure if needed
                if agent_name not in timing_store:
                    timing_store[agent_name] = {}
                if trial not in timing_store[agent_name]:
                    timing_store[agent_name][trial] = []

                timing_store[agent_name][trial].append(optimization_time)

            ## update the Phi_hat
            self.Phi_hat = Phi_main
            self.Phi_check = Phi_impute
        else:
            self.matching[self.t] = (
                (self.pseudo_action == self.chosen_action),
                None,
                None,
                None,
                None,
                None,
            )

    # beta is prev Phi
    def __imputation_loss(
        self, beta: np.ndarray, X: np.ndarray, Y: np.ndarray, r: np.ndarray, lam: float
    ):
        prev_impute = beta.reshape((self.M, self.N))
        loss = np.sum(np.power(r - np.einsum("ti,ij,tj->t", X, prev_impute, Y), 2))
        l1_norm = np.sum(np.abs(beta))
        return loss + (lam * l1_norm)

    # matching_history: (matched,x,y,pseudo_rewards,chosen_action,r,)
    def __main_loss(self, beta: np.ndarray, lam: float, matching_history: dict):
        # residuals_list = list()
        # for _, value in matching_history.items():
        #     if value[0]:
        #         residuals_list.append((value[3] - (np.kron(value[1],value[2])@beta)) ** 2)

        # # Sum all residuals efficiently
        # residuals_sum = sum(np.sum(residuals) for residuals in residuals_list)

        # # L1 regularization
        # l1_norm = np.sum(np.abs(beta))

        # # Total loss
        # return residuals_sum + lam * l1_norm

        # Extract matched keys and data
        matched_keys = [
            key for key, value in matching_history.items() if value[0]
        ]  # Filter matched entries

        X_list = [
            matching_history[key][1] for key in matched_keys
        ]  # List of X matrices

        Y_list = [
            matching_history[key][2] for key in matched_keys
        ]  # List of Y matrices

        pseudo_rewards_list = [
            matching_history[key][3] for key in matched_keys
        ]  # List of pseudo_rewards

        prev_main = beta.reshape((self.M, self.N))
        # Compute residuals for matched keys

        loss = np.sum(
            np.power(
                pseudo_rewards_list
                - np.einsum("tab,bc,tdc->tad", X_list, prev_main, Y_list),
                2,
            )
        )

        # residuals_list = [
        #     (pseudo_rewards - X @ prev_main @ Y.T) ** 2
        #     for X, Y, pseudo_rewards in zip(X_list, Y_list, pseudo_rewards_list)
        # ]

        # L1 regularization
        l1_norm = np.sum(np.abs(beta))

        # Total loss
        return loss + lam * l1_norm

    def __get_param(self):
        return {"param": self.Phi_hat, "impute": self.Phi_check}

class BiRoLFLasso(ContextualBandit):
    def __init__(
        self,
        M: int,
        N: int,
        sigma: float,
        delta: float,
        p: float,
        p1: float = None,
        p2: float = None,
        explore: bool = False,
        init_explore: int = 0,
        theoretical_init_explore: bool = False,
        lam_c_impute: float = 1.0,
        lam_c_main: float = 1.0,
        fista_max_iter: int = 200,
        fista_tol: float = 1e-6,
        kappa_cap: float = 0.0,
        kappa_cap_percentile: float = 0.0,
    ):
        # --- tunable regularization multipliers ---
        self.lam_c_impute = lam_c_impute
        self.lam_c_main = lam_c_main

        self.t = 0
        self.explore = explore
        self.init_explore = init_explore
        if theoretical_init_explore:
            pass

        self.M = M
        self.N = N
        self.delta = delta

        self.p = p
        self.p1 = p1 if p1 is not None else p
        self.p2 = p2 if p2 is not None else p
        self.sigma = sigma

        self.action_i_history = []
        self.action_j_history = []
        self.reward_history = []

        self.matching = dict()
        self.Phi_hat = np.zeros((self.M, self.N))
        self.Phi_check = np.zeros((self.M, self.N))
        self.impute_prev = np.zeros((self.M, self.N))
        self.main_prev = np.zeros((self.M, self.N))

        # ---- caches for fast proximal updates ----
        self._static_arms_initialized = False
        self.X_static = None   # (M, d_x)
        self.Y_static = None   # (N, d_y)

        # aggregates for impute objective
        self.Ncnt = np.zeros((self.M, self.N), dtype=int)
        self.Ssum = np.zeros((self.M, self.N), dtype=float)
        self.Ssq  = np.zeros((self.M, self.N), dtype=float)
        self.C_sum = np.zeros((self.M, self.N), dtype=float)

        # main-stage caches for fast full-matrix FISTA
        self.Gx = None
        self.Gy = None
        self.lam_x_max = None
        self.lam_y_max = None
        self.B_main_sum = np.zeros((self.M, self.N), dtype=float)
        self.Gamma_main = 0
        self._last_lam_main = None
        self.impute_use_backtracking = True

        # FISTA hyper-params
        self.fista_max_iter = int(fista_max_iter)
        self.fista_tol = float(fista_tol)
        self.kappa_cap = float(kappa_cap)
        self.kappa_cap_percentile = float(kappa_cap_percentile)
        self._arm_indices = np.arange(self.M * self.N)
        self._profile_ops = False
        self._kappa_x_cache = None
        self._kappa_y_cache = None

    # ---------- utilities ----------
    @staticmethod
    def _soft_threshold(Z: np.ndarray, tau: float) -> np.ndarray:
        return np.sign(Z) * np.maximum(np.abs(Z) - tau, 0.0)

    @staticmethod
    def _spectral_norm(M: np.ndarray, n_iter: int = 20) -> float:
        if M.size == 0:
            return 0.0
        v = np.random.randn(M.shape[1])
        v /= (np.linalg.norm(v) + 1e-12)
        for _ in range(n_iter):
            v = M.T @ (M @ v)
            nv = np.linalg.norm(v) + 1e-12
            v /= nv
        return float(np.sqrt(v @ (M.T @ (M @ v))))

    def _op_spectral_norm(self, op_func, shape, iters: int = 30, tol: float = 1e-6) -> float:
        x = np.random.randn(*shape)
        x /= (np.linalg.norm(x) + 1e-12)
        for _ in range(iters):
            y = op_func(x)
            ny = np.linalg.norm(y) + 1e-12
            x_next = y / ny
            if np.linalg.norm(x_next - x) < tol:
                return ny
            x = x_next
        return ny

    def _op_apply_main(self, W: np.ndarray, X_list: list, Y_list: list) -> np.ndarray:
        out = np.zeros_like(W)
        for Xt, Yt in zip(X_list, Y_list):
            out += Xt.T @ (Xt @ W @ Yt.T) @ Yt
        return out

    def _init_static_arms_if_needed(self, x: np.ndarray, y: np.ndarray) -> None:
        if self._static_arms_initialized:
            return
        self.X_static = x.copy()
        self.Y_static = y.copy()
        # Precompute Gram matrices for the main objective.
        self.Gx = self.X_static.T @ self.X_static
        self.Gy = self.Y_static.T @ self.Y_static
        self.lam_x_max = float(np.max(np.linalg.eigvalsh(self.Gx))) if self.Gx.size else 0.0
        self.lam_y_max = float(np.max(np.linalg.eigvalsh(self.Gy))) if self.Gy.size else 0.0
        # Cache capped kappa values once (x,y are fixed per trial).
        self._kappa_x_cache = self._compute_kappa(self.X_static)
        self._kappa_y_cache = self._compute_kappa(self.Y_static)
        self._static_arms_initialized = True

    def _update_impute_caches(self, i: int, j: int, r: float) -> None:
        self.Ncnt[i, j] += 1
        self.Ssum[i, j] += r
        self.Ssq[i, j]  += r * r
        self.C_sum += np.outer(self.X_static[i, :], self.Y_static[j, :]) * r

    def _compute_kappa(self, X: np.ndarray) -> float:
        if X.size == 0:
            return 0.0
        abs_x = np.abs(X)
        kappa = float(np.max(abs_x))
        cap = None
        if self.kappa_cap_percentile and self.kappa_cap_percentile > 0.0:
            pct = float(np.clip(self.kappa_cap_percentile, 0.0, 100.0))
            cap = float(np.percentile(abs_x, pct))
        if self.kappa_cap and self.kappa_cap > 0.0:
            cap = self.kappa_cap if cap is None else min(cap, self.kappa_cap)
        if cap is not None:
            kappa = min(kappa, cap)
        return kappa

    # ---- impute smooth part: g_imp(Φ) and ∇g_imp(Φ) ----
    def _g_impute(self, Phi: np.ndarray) -> float:
        if self.X_static is None or self.Y_static is None:
            return 0.0
        S = self.X_static @ Phi @ self.Y_static.T  # (M,N)
        term1 = float(np.sum(self.Ncnt * (S * S)))
        term2 = float(-2.0 * np.sum(self.Ssum * S))
        const = float(np.sum(self.Ssq))
        return term1 + term2 + const

    def _grad_impute(self, Phi: np.ndarray) -> np.ndarray:
        if self.X_static is None or self.Y_static is None:
            return np.zeros_like(Phi)
        S = self.X_static @ Phi @ self.Y_static.T
        Z = self.Ncnt * S
        return 2.0 * (self.X_static.T @ Z @ self.Y_static) - 2.0 * self.C_sum

    def _impute_lipschitz_upper(self) -> float:
        max_n = float(np.max(self.Ncnt)) if self.Ncnt.size else 0.0
        lam_x = max(float(self.lam_x_max or 0.0), 1e-12)
        lam_y = max(float(self.lam_y_max or 0.0), 1e-12)
        L = 2.0 * max_n * lam_x * lam_y
        return max(L, 1e-12)

    # ---- single, correct FISTA with backtracking (g = smooth only) ----
    def _fista_l1_backtracking(
        self,
        Phi0: np.ndarray,
        lam: float,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        g_fn: Callable[[np.ndarray], float],
        L0: float,
        gamma: float = 2.0,
        max_iter: int = None,
        tol: float = None,
        max_backtrack: int = 50,
        stats: Dict[str, float] = None,
    ) -> np.ndarray:
        if max_iter is None:
            max_iter = self.fista_max_iter
        if tol is None:
            tol = self.fista_tol

        L = max(L0, 1e-12)
        Phi = Phi0.copy()
        Y = Phi0.copy()
        t_par = 1.0

        iters = 0
        total_backtracks = 0
        converged = False
        for _ in range(max_iter):
            iters += 1
            G = grad_fn(Y)
            if not np.all(np.isfinite(G)):
                break

            gY = g_fn(Y)
            # backtracking line search on smooth part
            bt = 0
            while True:
                Phi_try = self._soft_threshold(Y - (1.0 / L) * G, lam / L)
                diff = Phi_try - Y
                Q  = gY + np.sum(G * diff) + 0.5 * L * np.linalg.norm(diff, ord='fro')**2
                g_try = g_fn(Phi_try)  # smooth part only
                if np.isfinite(g_try) and (g_try <= Q + 1e-12):
                    break
                L *= gamma
                bt += 1
                if bt >= max_backtrack:
                    # 안전장치: 너무 보수적인 L로 수렴이 굼뜨는 경우 루프 탈출
                    break
            total_backtracks += bt

            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_par * t_par))
            Y_next = Phi_try + ((t_par - 1.0) / t_next) * (Phi_try - Phi)

            # convergence check
            ndiff = np.linalg.norm(Phi_try - Phi, ord='fro')
            nphi  = np.linalg.norm(Phi,     ord='fro')
            Phi, Y, t_par = Phi_try, Y_next, t_next
            if ndiff <= (tol * max(1.0, nphi)):
                converged = True
                break
        if stats is not None:
            stats["iters"] = iters
            stats["converged"] = converged
            stats["backtracks"] = total_backtracks
        return Phi

    # (옵션) plain FISTA는 남겨둠 — 사용하지 않아도 무방
    def _fista_l1(
        self,
        Phi0: np.ndarray,
        lam: float,
        grad_fn,
        L_bound: float,
        max_iter: int = None,
        tol: float = None,
        stats: Dict[str, float] = None,
    ) -> np.ndarray:
        if max_iter is None:
            max_iter = self.fista_max_iter
        if tol is None:
            tol = self.fista_tol
        L = max(L_bound, 1e-12)
        eta = 1.0 / L
        Y = Phi0.copy()
        Phi = Phi0.copy()
        t_par = 1.0
        iters = 0
        converged = False
        for _ in range(max_iter):
            iters += 1
            G = grad_fn(Y)
            if np.any(np.isnan(G)) or np.any(np.isinf(G)):
                break
            Phi_next = self._soft_threshold(Y - eta * G, lam * eta)
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_par * t_par))
            Y = Phi_next + ((t_par - 1.0) / t_next) * (Phi_next - Phi)
            ndiff = np.linalg.norm(Phi_next - Phi, ord='fro')
            nphi  = np.linalg.norm(Phi,     ord='fro')
            if ndiff <= tol * max(1.0, nphi):
                Phi = Phi_next
                converged = True
                break
            Phi = Phi_next
            t_par = t_next
        if stats is not None:
            stats["iters"] = iters
            stats["converged"] = converged
        return Phi

    # ---------- policy ----------
    def choose(self, x: np.ndarray, y: np.ndarray):
        self.t += 1

        # greedy on current estimate (with optional warmup explore)
        if self.explore and (self.t <= self.init_explore):
            a_hat = np.random.choice(np.arange(self.M * self.N))
        else:
            decision = x @ self.Phi_hat @ y.T  # (M,N)
            a_hat = int(np.argmax(decision))

        i_hat, j_hat = action_to_ij(a_hat, self.N)
        self.a_hat = a_hat
        self.i_hat = i_hat
        self.j_hat = j_hat

        # store only once
        self._hat_history = getattr(self, "_hat_history", {})
        self._hat_history[self.t] = (i_hat, j_hat)

        # pseudo / chosen sampling (resample chosen and pseudo until match or max_iter)
        total_arms = self.M * self.N
        denom = np.log(1.0 / max(1.0 - self.p, 1e-12))
        max_iter = int(
            np.log(2.0 * ((self.t + 1) ** 2) / self.delta) / max(denom, 1e-12)
        )

        chosen_dist = np.full(total_arms, (1.0 / np.sqrt(self.t)) / max(total_arms - 1, 1), dtype=float)
        chosen_dist[a_hat] = 1 - (1.0 / np.sqrt(self.t))

        pseudo_action = -1
        chosen_action = -2
        count = 0
        while (pseudo_action != chosen_action) and (count <= max_iter):
            chosen_action = np.random.choice(self._arm_indices, p=chosen_dist).item()
            pseudo_dist = np.full(total_arms, (1.0 - self.p) / max(total_arms - 1, 1), dtype=float)
            pseudo_dist[chosen_action] = self.p
            pseudo_action = np.random.choice(self._arm_indices, p=pseudo_dist).item()
            count += 1

        self.pseudo_action = pseudo_action
        self.chosen_action = chosen_action

        return self.chosen_action

    def update(self, x: np.ndarray, y: np.ndarray, r: float):
        self._init_static_arms_if_needed(x, y)
        self._last_impute_time = 0.0
        self._last_main_time = 0.0
        self._last_impute_iters = 0
        self._last_main_iters = 0

        ci, cj = action_to_ij(self.chosen_action, self.N)

        # --- lambdas (BiRoLF-Lasso와 동일 스케일) ---
        kappa_x = self._kappa_x_cache if self._kappa_x_cache is not None else self._compute_kappa(x)
        kappa_y = self._kappa_y_cache if self._kappa_y_cache is not None else self._compute_kappa(y)
        lam_impute = self.lam_c_impute * (
            2 * self.sigma * kappa_x * kappa_y *
            np.sqrt(2 * self.t * np.log(2 * self.M * self.N * (self.t ** 2) / self.delta))
        )
        lam_main = self.lam_c_main * (
            (4 * self.sigma * kappa_x * kappa_y / self.p) *
            np.sqrt(2 * self.t * np.log(2 * self.M * self.N * (self.t ** 2) / self.delta))
        )
        self._last_lam_main = lam_main
        self._last_lam_main = lam_main

        if self.pseudo_action == self.chosen_action:
            self.reward_history.append(r)
            self._update_impute_caches(ci, cj, r)
            # ---- imputation: backtracking FISTA on smooth g_imp + l1 ----
            L_imp = self._impute_lipschitz_upper()
            impute_stats = {} if getattr(self, "_profile_ops", False) else None
            impute_start = time.perf_counter()
            if self.impute_use_backtracking:
                Phi_impute = self._fista_l1_backtracking(
                    self.impute_prev,
                    lam_impute,
                    self._grad_impute,
                    self._g_impute,
                    L_imp,
                    stats=impute_stats,
                )
            else:
                Phi_impute = self._fista_l1(
                    self.impute_prev,
                    lam_impute,
                    self._grad_impute,
                    L_imp,
                    stats=impute_stats,
                )
            self._last_impute_time = time.perf_counter() - impute_start
            if impute_stats is not None:
                self._last_impute_iters = int(impute_stats.get("iters", 0))

            # current round pseudo-reward contribution to main sufficient stats
            pred = float(x[ci, :] @ Phi_impute @ y[cj, :].T)
            alpha = (1.0 / max(self.p, 1e-12)) * (r - pred)
            self.B_main_sum += (self.Gx @ Phi_impute @ self.Gy) + alpha * np.outer(
                self.X_static[ci, :], self.Y_static[cj, :]
            )
            self.matching[self.t] = (True, None, None, None, self.chosen_action, r)
            self.Gamma_main += 1
            Gx_scaled = self.Gx * float(self.Gamma_main)
            B_main = self.B_main_sum
            L_main = 2.0 * max(self.lam_x_max, 1e-12) * max(self.lam_y_max, 1e-12) * float(self.Gamma_main)
            main_stats = {} if getattr(self, "_profile_ops", False) else None
            optimization_start_time = time.perf_counter()
            Phi_main = fista_lasso_matrix(
                Gx_scaled,
                self.Gy,
                B_main,
                lam_main,
                self.main_prev,
                L_main,
                self.fista_max_iter,
                self.fista_tol,
                use_fista=True,
                stats=main_stats,
            )
            optimization_time = time.perf_counter() - optimization_start_time
            self._last_main_time = optimization_time
            if main_stats is not None:
                self._last_main_iters = int(main_stats.get("iters", 0))

            if not getattr(self, "_benchmark_mode", False) and hasattr(self, '_timing_data') and self._timing_data is not None:
                agent = self.__class__.__name__
                trial = getattr(self, '_trial', 0)
                timing_store = self._timing_data.get("optimization", self._timing_data)
                timing_store.setdefault(agent, {}).setdefault(trial, []).append(optimization_time)

            # update params / warm-starts
            self.Phi_hat = Phi_main
            self.Phi_check = Phi_impute
            self.impute_prev = Phi_impute
            self.main_prev = Phi_main
        else:
            # unmatched: record only
            self.matching[self.t] = (False, None, None, None, None, None)

    # (옵션) 기존 loss helpers — 사용 안 해도 무관
    def __imputation_loss(self, beta: np.ndarray, X: np.ndarray, Y: np.ndarray, r: np.ndarray, lam: float):
        prev = beta.reshape((self.M, self.N))
        loss = np.sum((r - np.einsum("ti,ij,tj->t", X, prev, Y)) ** 2)
        return loss + lam * np.sum(np.abs(beta))

    def __main_loss(self, beta: np.ndarray, lam: float, matching_history: dict):
        matched_keys = [k for k, v in matching_history.items() if v[0]]
        X_list = [matching_history[k][1] for k in matched_keys]
        Y_list = [matching_history[k][2] for k in matched_keys]
        R_list = [matching_history[k][3] for k in matched_keys]
        Phi = beta.reshape((self.M, self.N))
        loss = 0.0
        for X, Y, R in zip(X_list, Y_list, R_list):
            loss += float(np.sum((R - X @ Phi @ Y.T) ** 2))
        return loss + lam * np.sum(np.abs(beta))

    def __get_param(self):
        return {"param": self.Phi_hat, "impute": self.Phi_check}

    def main_kkt_violation(self) -> float:
        """
        Return KKT residual for the last main update (0 means optimal up to tolerance).
        """
        if self.Gamma_main <= 0 or self.Gx is None or self.Gy is None:
            return 0.0
        if self._last_lam_main is None:
            return 0.0
        Gx_scaled = self.Gx * float(self.Gamma_main)
        return kkt_residual_matrix(Gx_scaled, self.Gy, self.B_main_sum, self.Phi_hat, self._last_lam_main)


def soft_threshold(Z: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(Z) * np.maximum(np.abs(Z) - tau, 0.0)


def fista_lasso_vector(
    G: np.ndarray,
    b: np.ndarray,
    mu: float,
    x0: np.ndarray,
    L: float,
    max_iter: int,
    tol: float,
    use_fista: bool = True,
    stats: Dict[str, float] = None,
) -> np.ndarray:
    assert G.shape[0] == G.shape[1]
    d = G.shape[0]
    b = np.asarray(b).reshape(-1)
    assert b.shape[0] == d
    if x0 is None:
        x = np.zeros(d, dtype=float)
    else:
        x = np.asarray(x0).reshape(-1).copy()
        assert x.shape[0] == d

    if d == 0:
        return x

    L = max(float(L), 1e-12)
    y = x.copy()
    t_par = 1.0
    iters = 0
    converged = False
    for _ in range(max_iter):
        iters += 1
        grad = 2.0 * (G @ y) - 2.0 * b
        x_next = soft_threshold(y - grad / L, mu / L)
        if use_fista:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_par * t_par))
            y = x_next + ((t_par - 1.0) / t_next) * (x_next - x)
            t_par = t_next
        else:
            y = x_next
        ndiff = np.linalg.norm(x_next - x)
        nrm = np.linalg.norm(x)
        x = x_next
        if ndiff <= tol * max(1.0, nrm):
            converged = True
            break
    if stats is not None:
        stats["iters"] = iters
        stats["converged"] = converged
    return x


def fista_lasso_matrix(
    Gx: np.ndarray,
    Gy: np.ndarray,
    B: np.ndarray,
    mu: float,
    X0: np.ndarray,
    L: float,
    max_iter: int,
    tol: float,
    use_fista: bool = True,
    stats: Dict[str, float] = None,
) -> np.ndarray:
    assert Gx.shape[0] == Gx.shape[1]
    assert Gy.shape[0] == Gy.shape[1]
    dx = Gx.shape[0]
    dy = Gy.shape[0]
    assert B.shape == (dx, dy)
    if X0 is None:
        X = np.zeros_like(B)
    else:
        X = np.asarray(X0).copy()
        assert X.shape == (dx, dy)

    if dx == 0 or dy == 0:
        return X

    L = max(float(L), 1e-12)
    Y = X.copy()
    t_par = 1.0
    iters = 0
    converged = False
    for _ in range(max_iter):
        iters += 1
        grad = 2.0 * (Gx @ Y @ Gy) - 2.0 * B
        X_next = soft_threshold(Y - grad / L, mu / L)
        if use_fista:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_par * t_par))
            Y = X_next + ((t_par - 1.0) / t_next) * (X_next - X)
            t_par = t_next
        else:
            Y = X_next
        ndiff = np.linalg.norm(X_next - X, ord="fro")
        nrm = np.linalg.norm(X, ord="fro")
        X = X_next
        if ndiff <= tol * max(1.0, nrm):
            converged = True
            break
    if stats is not None:
        stats["iters"] = iters
        stats["converged"] = converged
    return X


def kkt_residual_matrix(
    Gx: np.ndarray,
    Gy: np.ndarray,
    B: np.ndarray,
    Phi: np.ndarray,
    mu: float,
    eps: float = 1e-12,
) -> float:
    """
    KKT residual for matrix lasso:
      min tr(Phi^T Gx Phi Gy) - 2 tr(B^T Phi) + mu ||Phi||_1
    """
    if Phi.size == 0:
        return 0.0
    grad = 2.0 * (Gx @ Phi @ Gy - B)
    mask = np.abs(Phi) > eps
    res_nz = 0.0
    if np.any(mask):
        res_nz = float(np.max(np.abs(grad[mask] + mu * np.sign(Phi[mask]))))
    res_z = 0.0
    if np.any(~mask):
        res_z = float(np.max(np.maximum(np.abs(grad[~mask]) - mu, 0.0)))
    return max(res_nz, res_z)


# Batched ou solver: equivalent to independent column solves, but uses BLAS G @ X per step.
def fista_lasso_left_batched(
    G: np.ndarray,
    B: np.ndarray,
    mu: float,
    X0: np.ndarray,
    L: float,
    max_iter: int,
    tol: float,
    use_fista: bool = True,
    stats: Dict[str, float] = None,
) -> np.ndarray:
    assert G.shape[0] == G.shape[1]
    if X0 is None:
        X = np.zeros_like(B)
    else:
        X = np.asarray(X0).copy()
        assert X.shape == B.shape

    if B.size == 0:
        return X

    L = max(float(L), 1e-12)
    Y = X.copy()
    t_par = 1.0
    iters = 0
    converged = False
    for _ in range(max_iter):
        iters += 1
        grad = 2.0 * (G @ Y) - 2.0 * B
        X_next = soft_threshold(Y - grad / L, mu / L)
        if use_fista:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_par * t_par))
            Y = X_next + ((t_par - 1.0) / t_next) * (X_next - X)
            t_par = t_next
        else:
            Y = X_next
        ndiff = np.linalg.norm(X_next - X, ord="fro")
        nrm = np.linalg.norm(X, ord="fro")
        X = X_next
        if ndiff <= tol * max(1.0, nrm):
            converged = True
            break
    if stats is not None:
        stats["iters"] = iters
        stats["converged"] = converged
    return X


# Batched uo solver: equivalent to independent row solves, but uses BLAS X @ G per step.
def fista_lasso_right_batched(
    G: np.ndarray,
    B: np.ndarray,
    mu: float,
    X0: np.ndarray,
    L: float,
    max_iter: int,
    tol: float,
    use_fista: bool = True,
    stats: Dict[str, float] = None,
) -> np.ndarray:
    assert G.shape[0] == G.shape[1]
    if X0 is None:
        X = np.zeros_like(B)
    else:
        X = np.asarray(X0).copy()
        assert X.shape == B.shape

    if B.size == 0:
        return X

    L = max(float(L), 1e-12)
    Y = X.copy()
    t_par = 1.0
    iters = 0
    converged = False
    for _ in range(max_iter):
        iters += 1
        grad = 2.0 * (Y @ G) - 2.0 * B
        X_next = soft_threshold(Y - grad / L, mu / L)
        if use_fista:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_par * t_par))
            Y = X_next + ((t_par - 1.0) / t_next) * (X_next - X)
            t_par = t_next
        else:
            Y = X_next
        ndiff = np.linalg.norm(X_next - X, ord="fro")
        nrm = np.linalg.norm(X, ord="fro")
        X = X_next
        if ndiff <= tol * max(1.0, nrm):
            converged = True
            break
    if stats is not None:
        stats["iters"] = iters
        stats["converged"] = converged
    return X


def build_augmented_gram(G_obs: np.ndarray, size: int, d_obs: int) -> np.ndarray:
    assert 0 <= d_obs <= size
    G = np.eye(size, dtype=float)
    if d_obs > 0:
        G[:d_obs, :d_obs] = G_obs
    return G


def main_objective(Phi: np.ndarray, Gx: np.ndarray, Gy: np.ndarray, B: np.ndarray, mu: float) -> float:
    assert Phi.shape == B.shape
    if Phi.size == 0:
        return 0.0
    smooth = float(np.trace(Phi.T @ Gx @ Phi @ Gy) - 2.0 * np.trace(B.T @ Phi))
    l1 = float(mu) * float(np.sum(np.abs(Phi)))
    return smooth + l1


def main_objective_blockwise(
    Phi: np.ndarray,
    B: np.ndarray,
    Gx_o: np.ndarray,
    Gy_o: np.ndarray,
    dx: int,
    dy: int,
    mu: float,
) -> float:
    M, N = B.shape
    Gx = build_augmented_gram(Gx_o, M, dx)
    Gy = build_augmented_gram(Gy_o, N, dy)
    return main_objective(Phi, Gx, Gy, B, mu)


def solve_main_blockwise(
    B: np.ndarray,
    Gx_o: np.ndarray,
    Gy_o: np.ndarray,
    mu: float,
    dx: int,
    dy: int,
    Phi_init: np.ndarray = None,
    params: Dict[str, float] = None,
    lam_x_max: float = None,
    lam_y_max: float = None,
    stats: Dict[str, float] = None,
) -> np.ndarray:
    assert B.ndim == 2
    M, N = B.shape
    assert 0 <= dx <= M
    assert 0 <= dy <= N
    if Phi_init is None:
        Phi_init = np.zeros_like(B)
    else:
        assert Phi_init.shape == (M, N)

    if mu < 0.0:
        mu = 0.0

    if params is None:
        params = {}

    block_oo_max_iter = int(params.get("block_oo_max_iter", 100))
    block_ou_max_iter = int(params.get("block_ou_max_iter", 50))
    block_uo_max_iter = int(params.get("block_uo_max_iter", 50))
    block_tol = float(params.get("block_tol", 1e-6))
    block_use_fista = bool(params.get("block_use_fista", True))
    block_use_batched = bool(params.get("block_use_batched", True))

    Mu = M - dx
    Nu = N - dy

    if lam_x_max is None:
        lam_x_max = float(np.max(np.linalg.eigvalsh(Gx_o))) if dx > 0 else 0.0
    if lam_y_max is None:
        lam_y_max = float(np.max(np.linalg.eigvalsh(Gy_o))) if dy > 0 else 0.0

    # Blockwise closed-form (uu) + FISTA subproblems (oo/ou/uo).
    Phi_hat = np.zeros_like(B)
    B_oo = B[:dx, :dy]
    B_ou = B[:dx, dy:]
    B_uo = B[dx:, :dy]
    B_uu = B[dx:, dy:]
    P_oo = Phi_init[:dx, :dy]
    P_ou = Phi_init[:dx, dy:]
    P_uo = Phi_init[dx:, :dy]

    if Mu > 0 and Nu > 0:
        Phi_hat[dx:, dy:] = soft_threshold(B_uu, mu / 2.0)

    if dx > 0 and Nu > 0:
        Lx = 2.0 * max(lam_x_max, 1e-12)
        ou_stats = {} if stats is not None else None
        if block_use_batched:
            Phi_hat[:dx, dy:] = fista_lasso_left_batched(
                Gx_o,
                B_ou,
                mu,
                P_ou,
                Lx,
                block_ou_max_iter,
                block_tol,
                use_fista=block_use_fista,
                stats=ou_stats,
            )
        else:
            ou_iters_total = 0
            for k in range(Nu):
                col_stats = {} if stats is not None else None
                Phi_hat[:dx, dy + k] = fista_lasso_vector(
                    Gx_o,
                    B_ou[:, k],
                    mu,
                    P_ou[:, k],
                    Lx,
                    block_ou_max_iter,
                    block_tol,
                    use_fista=block_use_fista,
                    stats=col_stats,
                )
                if col_stats is not None:
                    ou_iters_total += int(col_stats.get("iters", 0))
            if stats is not None:
                stats["ou_iters"] = ou_iters_total
        if stats is not None and ou_stats is not None and "ou_iters" not in stats:
            stats["ou_iters"] = ou_stats.get("iters", 0)

    if Mu > 0 and dy > 0:
        Ly = 2.0 * max(lam_y_max, 1e-12)
        uo_stats = {} if stats is not None else None
        if block_use_batched:
            Phi_hat[dx:, :dy] = fista_lasso_right_batched(
                Gy_o,
                B_uo,
                mu,
                P_uo,
                Ly,
                block_uo_max_iter,
                block_tol,
                use_fista=block_use_fista,
                stats=uo_stats,
            )
        else:
            uo_iters_total = 0
            for l in range(Mu):
                row_stats = {} if stats is not None else None
                Phi_hat[dx + l, :dy] = fista_lasso_vector(
                    Gy_o,
                    B_uo[l, :],
                    mu,
                    P_uo[l, :],
                    Ly,
                    block_uo_max_iter,
                    block_tol,
                    use_fista=block_use_fista,
                    stats=row_stats,
                )
                if row_stats is not None:
                    uo_iters_total += int(row_stats.get("iters", 0))
            if stats is not None:
                stats["uo_iters"] = uo_iters_total
        if stats is not None and uo_stats is not None and "uo_iters" not in stats:
            stats["uo_iters"] = uo_stats.get("iters", 0)

    if dx > 0 and dy > 0:
        Loo = 2.0 * max(lam_x_max, 1e-12) * max(lam_y_max, 1e-12)
        oo_stats = {} if stats is not None else None
        Phi_hat[:dx, :dy] = fista_lasso_matrix(
            Gx_o,
            Gy_o,
            B_oo,
            mu,
            P_oo,
            Loo,
            block_oo_max_iter,
            block_tol,
            use_fista=block_use_fista,
            stats=oo_stats,
        )
        if stats is not None and oo_stats is not None:
            stats["oo_iters"] = oo_stats.get("iters", 0)

    if stats is not None:
        stats["total_iters"] = int(stats.get("ou_iters", 0) + stats.get("uo_iters", 0) + stats.get("oo_iters", 0))
    return Phi_hat


class BiRoLFLasso_Blockwise(BiRoLFLasso):
    def __init__(
        self,
        M: int,
        N: int,
        d_x: int,
        d_y: int,
        sigma: float,
        delta: float,
        p: float,
        p1: float = None,
        p2: float = None,
        explore: bool = False,
        init_explore: int = 0,
        theoretical_init_explore: bool = False,
        lam_c_impute: float = 1.0,
        lam_c_main: float = 1.0,
        fista_max_iter: int = 200,
        fista_tol: float = 1e-6,
        kappa_cap: float = 0.0,
        kappa_cap_percentile: float = 0.0,
        block_oo_max_iter: int = 100,
        block_ou_max_iter: int = 50,
        block_uo_max_iter: int = 50,
        block_tol: float = 1e-6,
        block_use_fista: bool = True,
        block_use_batched: bool = True,
    ):
        super().__init__(
            M=M,
            N=N,
            sigma=sigma,
            delta=delta,
            p=p,
            p1=p1,
            p2=p2,
            explore=explore,
            init_explore=init_explore,
            theoretical_init_explore=theoretical_init_explore,
            lam_c_impute=lam_c_impute,
            lam_c_main=lam_c_main,
            fista_max_iter=fista_max_iter,
            fista_tol=fista_tol,
            kappa_cap=kappa_cap,
            kappa_cap_percentile=kappa_cap_percentile,
        )
        self.dx = int(d_x)
        self.dy = int(d_y)
        assert 0 <= self.dx <= self.M
        assert 0 <= self.dy <= self.N

        self.block_oo_max_iter = block_oo_max_iter
        self.block_ou_max_iter = block_ou_max_iter
        self.block_uo_max_iter = block_uo_max_iter
        self.block_tol = block_tol
        self.block_use_fista = block_use_fista
        self.block_use_batched = block_use_batched

        self.Gamma = 0
        self._store_C = False
        self.C = np.zeros((self.M, self.N), dtype=float)
        self.B = np.zeros((self.M, self.N), dtype=float)
        self.G_Xo = None
        self.G_Yo = None
        self.lam_x_max = None
        self.lam_y_max = None
        self.kappa_x = None
        self.kappa_y = None
        self._block_caches_ready = False
        self.impute_use_backtracking = True
        self._profile_ops = False
        self._last_lam_main = None

    def _init_blockwise_caches(self, x: np.ndarray, y: np.ndarray) -> None:
        if self._block_caches_ready:
            return
        self._init_static_arms_if_needed(x, y)
        assert x.shape[0] == self.M and x.shape[1] == self.M
        assert y.shape[0] == self.N and y.shape[1] == self.N

        if self.dx > 0:
            x_obs = x[:, :self.dx]
            self.G_Xo = x_obs.T @ x_obs
            self.lam_x_max = float(np.max(np.linalg.eigvalsh(self.G_Xo)))
        else:
            self.G_Xo = np.zeros((0, 0), dtype=float)
            self.lam_x_max = 0.0

        if self.dy > 0:
            y_obs = y[:, :self.dy]
            self.G_Yo = y_obs.T @ y_obs
            self.lam_y_max = float(np.max(np.linalg.eigvalsh(self.G_Yo)))
        else:
            self.G_Yo = np.zeros((0, 0), dtype=float)
            self.lam_y_max = 0.0

        # Paper: kappa_max is max absolute entry in augmented features (optionally capped).
        self.kappa_x = self._compute_kappa(x)
        self.kappa_y = self._compute_kappa(y)
        self._block_caches_ready = True

    def update(self, x: np.ndarray, y: np.ndarray, r: float):
        self._init_blockwise_caches(x, y)
        self._last_impute_time = 0.0
        self._last_main_time = 0.0
        self._last_impute_iters = 0
        self._last_main_iters = 0
        self._last_block_iters = {}

        ci, cj = action_to_ij(self.chosen_action, self.N)

        kappa_x = self.kappa_x if self.kappa_x is not None else np.max(np.abs(x))
        kappa_y = self.kappa_y if self.kappa_y is not None else np.max(np.abs(y))
        lam_impute = self.lam_c_impute * (
            2 * self.sigma * kappa_x * kappa_y *
            np.sqrt(2 * self.t * np.log(2 * self.M * self.N * (self.t ** 2) / self.delta))
        )
        lam_main = self.lam_c_main * (
            (4 * self.sigma * kappa_x * kappa_y / self.p) *
            np.sqrt(2 * self.t * np.log(2 * self.M * self.N * (self.t ** 2) / self.delta))
        )

        if self.pseudo_action != self.chosen_action:
            return

        self.reward_history.append(r)
        self._update_impute_caches(ci, cj, r)

        L_imp = self._impute_lipschitz_upper()
        impute_stats = {} if getattr(self, "_profile_ops", False) else None
        impute_start = time.perf_counter()
        if self.impute_use_backtracking:
            Phi_impute = self._fista_l1_backtracking(
                self.impute_prev,
                lam_impute,
                self._grad_impute,
                self._g_impute,
                L_imp,
                stats=impute_stats,
            )
        else:
            Phi_impute = self._fista_l1(
                self.impute_prev,
                lam_impute,
                self._grad_impute,
                L_imp,
                stats=impute_stats,
            )
        self._last_impute_time = time.perf_counter() - impute_start
        if impute_stats is not None:
            self._last_impute_iters = int(impute_stats.get("iters", 0))

        pred = float(x[ci, :] @ Phi_impute @ y[cj, :].T)
        # Conditional pseudo sampling -> constant 1/p correction for unbiasedness.
        w_now = 1.0 / max(self.p, 1e-12)
        alpha = w_now * (r - pred)

        # Rank-1 correction per matched round without materializing full R_t.
        dx, dy = self.dx, self.dy
        Phi = Phi_impute
        Phi_oo = Phi[:dx, :dy]
        Phi_ou = Phi[:dx, dy:]
        Phi_uo = Phi[dx:, :dy]
        Phi_uu = Phi[dx:, dy:]

        T_oo = self.G_Xo @ Phi_oo @ self.G_Yo if dx > 0 and dy > 0 else np.zeros((dx, dy))
        T_ou = self.G_Xo @ Phi_ou if dx > 0 else np.zeros((dx, self.N - dy))
        T_uo = Phi_uo @ self.G_Yo if dy > 0 else np.zeros((self.M - dx, dy))
        T_uu = Phi_uu if (self.M - dx) > 0 and (self.N - dy) > 0 else np.zeros((self.M - dx, self.N - dy))

        x_o = x[ci, :dx]
        x_u = x[ci, dx:]
        y_o = y[cj, :dy]
        y_u = y[cj, dy:]
        O_oo = np.outer(x_o, y_o)
        O_ou = np.outer(x_o, y_u)
        O_uo = np.outer(x_u, y_o)
        O_uu = np.outer(x_u, y_u)

        S_oo = T_oo + alpha * O_oo
        S_ou = T_ou + alpha * O_ou
        S_uo = T_uo + alpha * O_uo
        S_uu = T_uu + alpha * O_uu

        # Online averages for B_t without full-matrix division.
        self.Gamma += 1
        inv_gamma = 1.0 / float(self.Gamma)
        self.B[:dx, :dy] += (S_oo - self.B[:dx, :dy]) * inv_gamma
        self.B[:dx, dy:] += (S_ou - self.B[:dx, dy:]) * inv_gamma
        self.B[dx:, :dy] += (S_uo - self.B[dx:, :dy]) * inv_gamma
        self.B[dx:, dy:] += (S_uu - self.B[dx:, dy:]) * inv_gamma
        if self._store_C:
            self.C[:dx, :dy] += S_oo
            self.C[:dx, dy:] += S_ou
            self.C[dx:, :dy] += S_uo
            self.C[dx:, dy:] += S_uu

        mu = lam_main / float(self.Gamma)
        main_stats = {} if getattr(self, "_profile_ops", False) else None
        optimization_start_time = time.perf_counter()
        Phi_main = solve_main_blockwise(
            B=self.B,
            Gx_o=self.G_Xo,
            Gy_o=self.G_Yo,
            mu=mu,
            dx=dx,
            dy=dy,
            Phi_init=self.main_prev,
            params={
                "block_oo_max_iter": self.block_oo_max_iter,
                "block_ou_max_iter": self.block_ou_max_iter,
                "block_uo_max_iter": self.block_uo_max_iter,
                "block_tol": self.block_tol,
                "block_use_fista": self.block_use_fista,
                "block_use_batched": self.block_use_batched,
            },
            lam_x_max=self.lam_x_max,
            lam_y_max=self.lam_y_max,
            stats=main_stats,
        )
        optimization_time = time.perf_counter() - optimization_start_time
        self._last_main_time = optimization_time
        if main_stats is not None:
            self._last_main_iters = int(main_stats.get("total_iters", 0))
            self._last_block_iters = {
                "oo_iters": int(main_stats.get("oo_iters", 0)),
                "ou_iters": int(main_stats.get("ou_iters", 0)),
                "uo_iters": int(main_stats.get("uo_iters", 0)),
            }

        if not getattr(self, "_benchmark_mode", False) and hasattr(self, "_timing_data") and self._timing_data is not None:
            agent = self.__class__.__name__
            trial = getattr(self, "_trial", 0)
            timing_store = self._timing_data.get("optimization", self._timing_data)
            timing_store.setdefault(agent, {}).setdefault(trial, []).append(optimization_time)

        self.Phi_hat = Phi_main
        self.Phi_check = Phi_impute
        self.impute_prev = Phi_impute
        self.main_prev = Phi_main

    def main_kkt_violation(self) -> float:
        """
        KKT residual for blockwise main objective.
        """
        if self.Gamma <= 0 or self._last_lam_main is None:
            return 0.0
        Gx = build_augmented_gram(self.G_Xo, self.M, self.dx)
        Gy = build_augmented_gram(self.G_Yo, self.N, self.dy)
        mu = self._last_lam_main / float(self.Gamma)
        return kkt_residual_matrix(Gx, Gy, self.B, self.Phi_hat, mu)


class LowOFUL(ContextualBandit):
    """
    LowOFUL: A variant of OFUL for almost-low-dimensional structure.
    Implements Algorithm 1 from “Bilinear Bandits with Low-rank Structure”:
      - p: ambient dimension
      - k: “good” subspace dimension
      - lam: regularization on the first k coordinates
      - lam_perp: regularization on the remaining p–k coordinates
      - B: bound on ||θ*||₂
      - B_perp: bound on ||θ*_{k+1:p}||₂
      - delta: failure probability
      - sigma: noise scale (sub-Gaussian)
    """
    def __init__(
        self,
        p: int,
        k: int,
        lam: float,
        lam_perp: float,
        B: float,
        B_perp: float,
        delta: float,
        sigma: float,
    ):
        self.p = p
        self.k = k
        self.lam = lam
        self.lam_perp = lam_perp
        self.B = B
        self.B_perp = B_perp
        self.delta = delta
        self.sigma = sigma

        # Initialize Λ = diag([lam]*k, [lam_perp]*(p-k))
        # and its inverse Vinv = Λ⁻¹
        inv_diag = np.array([1/lam] * k + [1/lam_perp] * (p - k), dtype=float)
        self.Vinv = np.diag(inv_diag)

        # Store log-det of Λ for confidence radius updates
        self.logdet_Lambda = k * np.log(lam) + (p - k) * np.log(lam_perp)
        self.logdet_V = self.logdet_Lambda

        # Initialize xty = V⁻¹·Xᵀy accumulator and time counter
        self.xty = np.zeros(p)
        self.t = 0

        # Placeholders for last chosen context and estimate
        self.last_arm = None
        self.theta_hat = np.zeros(p)

    def choose(self, x: np.ndarray) -> int:
        """
        Select an arm via optimism in face of uncertainty (OFUL).
        x: array of shape (N, p) where each row is an arm's feature vector.
        Returns the index of the chosen arm.
        """
        self.t += 1

        # Compute ridge estimator θ̂_t = V⁻¹ · (Xᵀ y)
        self.theta_hat = self.Vinv @ self.xty

        # Compute confidence radius β_t
        log_ratio = self.logdet_V - self.logdet_Lambda
        radius = (
            self.sigma * np.sqrt(2 * np.log(np.exp(log_ratio) / (self.delta**2)))
            + np.sqrt(self.lam) * self.B
            + np.sqrt(self.lam_perp) * self.B_perp
        )

        # Compute UCB scores for each arm: <θ̂, x> + radius · ||x||_{V⁻¹}
        expected = x @ self.theta_hat  # shape (N,)
        widths = np.sqrt(np.einsum("ni,ij,nj->n", x, self.Vinv, x))
        ucb_scores = expected + radius * widths

        # Pick any argmax at random
        candidates = np.where(ucb_scores == np.max(ucb_scores))[0]
        self.chosen_action = np.random.choice(candidates)
        self.last_arm = x[self.chosen_action]
        return self.chosen_action

    def update(self, x: np.ndarray, r: float) -> None:
        """
        Update the model with observed reward.
        x: the same (N, p) matrix passed to choose (not used directly here).
        r: reward obtained from the chosen arm.
        """
        a = self.last_arm  # feature vector of the chosen arm

        # Update log-det V: det(V + aaᵀ) = det(V) · (1 + aᵀ V⁻¹ a)
        va = float(a @ self.Vinv @ a)
        self.logdet_V += np.log(1 + va)

        # Update V⁻¹ via Sherman–Morrison
        self.Vinv = shermanMorrison(self.Vinv, a)

        # Update xty accumulator
        self.xty += r * a

    def __get_param(self) -> Dict[str, np.ndarray]:
        return {"param": self.theta_hat}

class ESTRLowOFUL(ContextualBandit):
    """
    Explore-Subspace-Then-Refine (ESTR) + LowOFUL for bilinear bandits.

    Stage 1 (exploration & subspace estimation):
      - Select d1 and d2 well-conditioned arms from X and Z (approx via pivoted QR).
      - For T1 rounds, pull pairs from the d1×d2 grid nearly uniformly.
      - Build K_tilde with average rewards for each (i,j) in the grid.
      - Estimate Θ_hat = X_sel^{-1} · K_tilde · (Z_sel^T)^{-1}.
      - Compute SVD(Θ_hat) = U_hat S_hat V_hat^T and orthonormal complements U_perp_hat, V_perp_hat.

    Stage 2 (refinement):
      - Rotate arms by Qx=[U_hat U_perp_hat] and Qz=[V_hat V_perp_hat].
      - Vectorize each pair (x', z') into a d1*d2-dimensional feature a where the first
        k=(d1+d2)r-r^2 coordinates are from the signal blocks
        (x_r z_r^T, x_perp z_r^T, x_r z_perp^T), and the last (d1-r)(d2-r) coordinates
        are from the complementary block (x_perp z_perp^T).
      - Run LowOFUL over these a-vectors.

    Notes:
      * Comments are in English; user-facing explanation can be provided separately.
      * This class mirrors the BiRoLFLasso signature: choose(x: ndarray, y: ndarray) / update(x, y, r).
    """

    def __init__(
        self,
        d1: int,
        d2: int,
        r: int,
        T1: int,
        lam: float,
        lam_perp: float,
        B: float,
        B_perp: float,
        delta: float,
        sigma: float,
    ) -> None:
        # Dimensions and hyperparameters
        self.d1 = d1
        self.d2 = d2
        self.r = r
        self.T1 = T1

        # LowOFUL for Stage 2
        p = d1 * d2
        k = (d1 + d2) * r - (r * r)
        self.low_oful = LowOFUL(
            p=p,
            k=k,
            lam=lam,
            lam_perp=lam_perp,
            B=B,
            B_perp=B_perp,
            delta=delta,
            sigma=sigma,
        )

        # Stage bookkeeping
        self.t = 0
        self._stage2_ready = False

        # Stage-1 selections and accumulators
        self.X_sel_idx = None  # indices of selected d1 arms from X
        self.Z_sel_idx = None  # indices of selected d2 arms from Z
        self.X_sel = None      # (d1, d1)
        self.Z_sel = None      # (d2, d2)
        self.K_sum = None      # (d1, d2) sum of rewards
        self.K_cnt = None      # (d1, d2) counts

        # Subspace objects (set at end of Stage 1)
        self.U_hat = None
        self.V_hat = None
        self.U_perp_hat = None
        self.V_perp_hat = None
        self.Qx = None  # (d1, d1) = [U_hat U_perp_hat]
        self.Qz = None  # (d2, d2) = [V_hat V_perp_hat]

        # Cache for last chosen pair in Stage 1
        self._last_pair_stage1 = None  # (i_idx, j_idx) in selected grid

        # Mapping for Stage 2: index -> (i, j)
        self._pair_map = None

    # ---------- Stage 1 helpers ----------
    def _select_well_conditioned(self, X: np.ndarray, m: int) -> np.ndarray:
        """Pick m rows of X that are approximately well-conditioned using pivoted QR."""
        # Use QR with column pivoting on X^T to select rows.
        # We choose the row indices corresponding to largest leverage.
        from scipy.linalg import qr
        QT, R, piv = qr(X.T, mode='economic', pivoting=True)
        return np.array(piv[:m], dtype=int)

    def _init_stage1(self, X: np.ndarray, Z: np.ndarray) -> None:
        """Initialize Stage-1 selections and accumulators from the current arm sets."""
        if self.X_sel_idx is None:
            self.X_sel_idx = self._select_well_conditioned(X, self.d1)
            self.Z_sel_idx = self._select_well_conditioned(Z, self.d2)
            self.X_sel = X[self.X_sel_idx, :]  # (d1, d1)
            self.Z_sel = Z[self.Z_sel_idx, :]  # (d2, d2)
            self.K_sum = np.zeros((self.d1, self.d2), dtype=float)
            self.K_cnt = np.zeros((self.d1, self.d2), dtype=int)

    def _schedule_stage1_pair(self) -> tuple:
        """Choose the next (i,j) within the d1×d2 grid to balance counts."""
        # Pull pairs as evenly as possible; break ties randomly.
        min_cnt = np.min(self.K_cnt)
        candidates = np.argwhere(self.K_cnt == min_cnt)
        i, j = candidates[np.random.randint(len(candidates))]
        return int(i), int(j)

    def _finalize_stage1(self) -> None:
        """Estimate Θ_hat via the d1×d2 averages and compute subspaces."""
        # Average reward matrix on the selected grid
        K_tilde = np.divide(
            self.K_sum,
            np.maximum(self.K_cnt, 1),
            out=np.zeros_like(self.K_sum, dtype=float),
            where=(self.K_cnt > 0),
        )
        # Θ_hat = X^{-1} K_tilde (Z^T)^{-1}
        X_inv = np.linalg.pinv(self.X_sel)
        Z_inv_T = np.linalg.pinv(self.Z_sel.T)
        Theta_hat = X_inv @ K_tilde @ Z_inv_T

        # SVD and orthogonal complements
        U, S, Vt = np.linalg.svd(Theta_hat, full_matrices=True)
        self.U_hat = U[:, : self.r]
        self.V_hat = Vt.T[:, : self.r]

        # Orthonormal complements (span(U_hat)^⊥ and span(V_hat)^⊥)
        # Use QR to complete to an orthonormal basis
        Qx_full, _ = np.linalg.qr(np.concatenate([self.U_hat, np.eye(self.d1)], axis=1))
        Qz_full, _ = np.linalg.qr(np.concatenate([self.V_hat, np.eye(self.d2)], axis=1))
        self.U_perp_hat = Qx_full[:, self.r : self.r + (self.d1 - self.r)]
        self.V_perp_hat = Qz_full[:, self.r : self.r + (self.d2 - self.r)]

        # Rotation matrices
        self.Qx = np.concatenate([self.U_hat, self.U_perp_hat], axis=1)  # (d1,d1)
        self.Qz = np.concatenate([self.V_hat, self.V_perp_hat], axis=1)  # (d2,d2)

        self._stage2_ready = True

    # ---------- Stage 2 helpers ----------
    @staticmethod
    def _vecF(A: np.ndarray) -> np.ndarray:
        """Vectorize by columns (Fortran order) to mimic vec(·)."""
        return A.reshape(-1, order='F')

    def _build_A(self, X_rot: np.ndarray, Z_rot: np.ndarray) -> tuple:
        """
        Build the vectorized arm set A and a map from index to (i,j).
        X_rot: (M, d1)  Z_rot: (N, d2)
        Returns (A_matrix, pair_map)
        """
        M, N = X_rot.shape[0], Z_rot.shape[0]
        r = self.r
        d1, d2 = self.d1, self.d2
        k = (d1 + d2) * r - (r * r)
        p = d1 * d2

        A = np.zeros((M * N, p), dtype=float)
        pair_map = []

        for i in range(M):
            x = X_rot[i]
            x_r, x_perp = x[:r], x[r:]
            for j in range(N):
                z = Z_rot[j]
                z_r, z_perp = z[:r], z[r:]
                # Blocks
                block1 = self._vecF(np.outer(x_r, z_r))                 # r×r
                block2 = self._vecF(np.outer(x_perp, z_r))              # (d1-r)×r
                block3 = self._vecF(np.outer(x_r, z_perp))              # r×(d2-r)
                block4 = self._vecF(np.outer(x_perp, z_perp))           # (d1-r)×(d2-r)
                # Concatenate in the order used by ESTR
                a = np.concatenate([block1, block2, block3, block4], axis=0)
                A[i * N + j, :] = a
                pair_map.append((i, j))
        return A, pair_map

    # ---------- Public API ----------
    def choose(self, x: np.ndarray, y: np.ndarray):
        """
        x: (M, d1) user-side arms, each row is a feature vector
        y: (N, d2) item-side arms, each row is a feature vector
        Returns the encoded action index i*N + j (consistent with BiRoLFLasso).
        """
        self.t += 1
        # Stage 1 initialization
        self._init_stage1(x, y)

        if self.t <= self.T1:
            # Stage 1: explore the d1×d2 grid
            i_loc, j_loc = self._schedule_stage1_pair()
            i_global = int(self.X_sel_idx[i_loc])
            j_global = int(self.Z_sel_idx[j_loc])
            self._last_pair_stage1 = (i_loc, j_loc)
            return i_global * y.shape[0] + j_global

        # If we just finished Stage 1, finalize subspaces
        if not self._stage2_ready:
            self._finalize_stage1()

        # Stage 2: rotate arms and invoke LowOFUL
        X_rot = x @ self.Qx  # (M, d1)
        Y_rot = y @ self.Qz  # (N, d2)
        A, pair_map = self._build_A(X_rot, Y_rot)
        self._pair_map = pair_map

        idx = self.low_oful.choose(A)
        i, j = pair_map[idx]
        self._last_pair_stage2 = (i, j)
        return i * y.shape[0] + j

    def update(self, x: np.ndarray, y: np.ndarray, r: float):
        """Update algorithm with observed reward r for the previously chosen pair."""
        if self.t <= self.T1:
            # Accumulate for K_tilde
            i_loc, j_loc = self._last_pair_stage1
            self.K_sum[i_loc, j_loc] += r
            self.K_cnt[i_loc, j_loc] += 1
            return

        # Stage 2 update: just forward to LowOFUL
        # Rebuild A to conform to the current arm sets (choose() already cached pair_map)
        X_rot = x @ self.Qx
        Y_rot = y @ self.Qz
        A, _ = self._build_A(X_rot, Y_rot)
        self.low_oful.update(A, r)

    def __get_param(self):
        # Expose the subspace estimate and LowOFUL parameter
        out = {
            "U_hat": self.U_hat,
            "V_hat": self.V_hat,
            "U_perp_hat": self.U_perp_hat,
            "V_perp_hat": self.V_perp_hat,
        }
        out.update(self.low_oful.__get_param())
        return out
