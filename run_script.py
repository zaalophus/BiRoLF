from main import run_main
from cfg import get_cfg


# parser = argparse.ArgumentParser()
# parser.add_argument("--explore", action="store_true", default=True)
# parser.add_argument("--init_explore", type=str, choices=["quad", "triple", "sqr", "K", "double"], default="double")
# parser.add_argument("--verbose", action="store_true", default=False)
# parser.add_argument("--trials", type=int, default=1)
# parser.add_argument("--case", type=int, default=1)

# ## For Bilinear model
# parser.add_argument("--arm_x", "-M", type=int, default=5)
# parser.add_argument("--arm_y", "-N", type=int, default=5)
# parser.add_argument("--true_dim_x", "-tx", type=int, default=4)
# parser.add_argument("--true_dim_y", "-ty", type=int, default=4)
# parser.add_argument("--dim_x", "-dx", type=int, default=3)
# parser.add_argument("--dim_y", "-dy", type=int, default=3)
# parser.add_argument("--p1", type=float, default=None, help="(Optional) legacy BiRoLF p1; defaults to p")
# parser.add_argument("--p2", type=float, default=None, help="(Optional) legacy BiRoLF p2; defaults to p")

# parser.add_argument("--horizon", "-T", type=int, default=10)
# parser.add_argument("--seed", "-S", type=int, default=123)

# parser.add_argument("--reward_dist", "-RD", type=str, default="gaussian")
# parser.add_argument("--reward_std", "-RS", type=float, default=0.1)

# parser.add_argument("--param_dist", "-PD", type=str, default="uniform")
# parser.add_argument("--param_bound", "-PB", type=float, default=1.)
# parser.add_argument("--param_bound_type", "-PBT", type=str, choices=["l1", "l2", "lsup"])
# parser.add_argument("--param_uniform_rng", "-PUR", type=float, default=None, nargs=2)

# parser.add_argument("--filetype", type=str, choices=["pickle", "json"], default="pickle")
# parser.add_argument("--delta", type=float, default=0.1)
# parser.add_argument("--p", type=float, default=0.6)
# # parser.add_argument("--date", type=str, default=None)


# # --- Regularization scale multipliers (CLI-tunable) ---
# parser.add_argument("--lamc_rolf_impute", type=float, default=0.01,
#                     help="Scale for RoLFLasso imputation lambda")
# parser.add_argument("--lamc_rolf_main", type=float, default=0.01,
#                     help="Scale for RoLFLasso main lambda")
# parser.add_argument("--lamc_bi_impute", type=float, default=0.01,
#                     help="Scale for BiRoLF (Lasso/FISTA) imputation lambda")
# parser.add_argument("--lamc_bi_main", type=float, default=0.01,
#                     help="Scale for BiRoLF (Lasso/FISTA) main lambda")

# # --- Blockwise main solver options ---
# parser.add_argument("--optimizer_max_iter", type=int, default=100)
    
# parser.add_argument("--block_tol", type=float, default=1e-6)
# parser.add_argument("--block_use_fista", type=str2bool, default=True)
# parser.add_argument("--block_use_batched", type=str2bool, default=True)

# # --- Optional BLAS thread control for multiprocessing ---
# parser.add_argument("--set_blas_threads", type=str2bool, default=False)
# parser.add_argument("--blas_threads_per_worker", type=int, default=1)

class Script:
    def __init__(self, cfg,
                 explore= True,
                 init_explore="double",
                 trials=5,
                 case=1,
                 arm_x=5,
                 arm_y=5,
                 true_dim_x=4,
                 true_dim_y=4,
                 dim_x=3,
                 dim_y=3,
                 p1=None,
                 p2=None,
                 horizon=100,
                 seed=1,
                 reward_std=0.1,
                 delta=0.1,
                 p=0.6,
                 lamc_rolf_impute=0.01,
                 lamc_rolf_main=0.01,
                 lamc_bi_impute=0.01,
                 lamc_bi_main=0.01,
                 optimizer_max_iter = 1000,
                 block_tol=1e-6,
                 block_use_fista=True,
                 block_use_batched=True,
                 set_blas_threads=False,
                 blas_threads_per_worker=1):
        self.cfg = cfg
        self.cfg.explore= explore
        self.cfg.init_explore=init_explore
        self.cfg.trials=trials
        self.cfg.case=case
        self.cfg.arm_x=arm_x
        self.cfg.arm_y=arm_y
        self.cfg.true_dim_x=true_dim_x
        self.cfg.true_dim_y=true_dim_y
        self.cfg.dim_x=dim_x
        self.cfg.dim_y=dim_y
        self.cfg.p1=p1
        self.cfg.p2=p2
        self.cfg.horizon=horizon
        self.cfg.seed=seed
        self.cfg.reward_std = reward_std
        self.cfg.delta = delta
        self.cfg.p=p
        self.cfg.lamc_rolf_impute=lamc_rolf_impute
        self.cfg.lamc_rolf_main=lamc_rolf_main
        self.cfg.lamc_bi_impute=lamc_bi_impute
        self.cfg.lamc_bi_main=lamc_bi_main
        self.cfg.optimizer_max_iter = optimizer_max_iter
        self.cfg.block_tol=block_tol
        self.cfg.block_use_fista=block_use_fista
        self.cfg.block_use_batched=block_use_batched
        self.cfg.set_blas_threads=set_blas_threads
        self.cfg.blas_threads_per_worker=blas_threads_per_worker
        
        
# default case : true_dim_x − arm_x < dim_x < arm_x -> arm_x=10,  true_dim_x=14, dim_x=5 
if __name__ == "__main__":
    cfg = get_cfg()
    for seed in [23,29,31,37]:
        for case in range(1,10):
            for init_explore in ["double"]:
                now_script = Script(cfg,
                            explore= True,
                            init_explore=init_explore,
                            trials=5,
                            case=case,
                            arm_x=10,
                            arm_y=10,
                            true_dim_x=14,
                            true_dim_y=14,
                            dim_x=5,
                            dim_y=5,
                            p1=None,
                            p2=None,
                            horizon=700,
                            seed=seed,
                            reward_std=0.1,
                            delta=0.1,
                            p=0.6,
                            lamc_rolf_impute=0.01,
                            lamc_rolf_main=0.01,
                            lamc_bi_impute=0.01,
                            lamc_bi_main=0.01,
                            optimizer_max_iter = 1000)
                run_main(now_script.cfg)