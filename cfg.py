import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def tuple_type(inputs):
    return tuple(map(float, inputs.split(',')))

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explore", action="store_true", default=True)
    parser.add_argument("--init_explore", type=str, choices=["quad", "triple", "sqr", "K", "double"], default="double")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--case", type=int, default=1)

    ## For Linear model only
    # parser.add_argument("--arms", "-N", type=int, default=20)
    # parser.add_argument("--latent_dim", "-l", type=int, default=10)
    # parser.add_argument("--dim", "-d", type=int, default=10)
    
    ## For Bilinear model
    parser.add_argument("--arm_x", "-M", type=int, default=5)
    parser.add_argument("--arm_y", "-N", type=int, default=5)
    parser.add_argument("--true_dim_x", "-tx", type=int, default=4)
    parser.add_argument("--true_dim_y", "-ty", type=int, default=4)
    parser.add_argument("--dim_x", "-dx", type=int, default=3)
    parser.add_argument("--dim_y", "-dy", type=int, default=3)
    parser.add_argument("--p1", type=float, default=None, help="(Optional) legacy BiRoLF p1; defaults to p")
    parser.add_argument("--p2", type=float, default=None, help="(Optional) legacy BiRoLF p2; defaults to p")

    parser.add_argument("--horizon", "-T", type=int, default=1000)
    parser.add_argument("--seed", "-S", type=int, default=123)
    parser.add_argument("--rho_sq", type=float, default=0.5)
    
    parser.add_argument("--feat_dist", "-FD", type=str, default="gaussian")
    parser.add_argument("--feat_disjoint", action="store_true")
    parser.add_argument("--feat_cov_dist", "-FCD", type=str, default=None)
    parser.add_argument("--feat_uniform_rng", "-FUR", type=float, default=None, nargs=2)
    parser.add_argument("--feat_feature_bound", "-OFB", type=float, default=None)
    parser.add_argument("--feat_bound_method", "-OBM", type=str, choices=["scaling", "clipping"], default=None)
    parser.add_argument("--feat_bound_type", "-FBT", type=str, choices=["l1", "l2", "lsup"])
    
    parser.add_argument("--map_dist", "-MD", type=str, default="uniform")
    parser.add_argument("--map_lower_bound", "-MLB", type=float, default=None)
    parser.add_argument("--map_upper_bound", "-MUB", type=float, default=None)
    parser.add_argument("--map_uniform_rng", "-MUR", type=float, default=None, nargs=2)
    
    parser.add_argument("--reward_dist", "-RD", type=str, default="gaussian")
    parser.add_argument("--reward_std", "-RS", type=float, default=0.1)
    
    parser.add_argument("--param_dist", "-PD", type=str, default="uniform")
    parser.add_argument("--param_bound", "-PB", type=float, default=1.)
    parser.add_argument("--param_bound_type", "-PBT", type=str, choices=["l1", "l2", "lsup"])
    parser.add_argument("--param_uniform_rng", "-PUR", type=float, default=None, nargs=2)
    parser.add_argument("--param_disjoint", action="store_true")
    
    parser.add_argument("--save_param", action="store_true")
    parser.add_argument("--filetype", type=str, choices=["pickle", "json"], default="pickle")
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--p", type=float, default=0.6)
    # parser.add_argument("--date", type=str, default=None)
    

    # --- Regularization scale multipliers (CLI-tunable) ---
    parser.add_argument("--lamc_rolf_impute", type=float, default=0.01,
                        help="Scale for RoLFLasso imputation lambda")
    parser.add_argument("--lamc_rolf_main", type=float, default=0.01,
                        help="Scale for RoLFLasso main lambda")
    parser.add_argument("--lamc_bi_impute", type=float, default=0.01,
                        help="Scale for BiRoLF (Lasso/FISTA) imputation lambda")
    parser.add_argument("--lamc_bi_main", type=float, default=0.01,
                        help="Scale for BiRoLF (Lasso/FISTA) main lambda")

    # --- Blockwise main solver options ---
    parser.add_argument("--block_oo_max_iter", type=int, default=100)
    parser.add_argument("--block_ou_max_iter", type=int, default=50)
    parser.add_argument("--block_uo_max_iter", type=int, default=50)
    parser.add_argument("--block_tol", type=float, default=1e-6)
    parser.add_argument("--block_use_fista", type=str2bool, default=True)

    # --- Optional BLAS thread control for multiprocessing ---
    parser.add_argument("--set_blas_threads", type=str2bool, default=False)
    parser.add_argument("--blas_threads_per_worker", type=int, default=1)

    cfg = parser.parse_args()
    if cfg.p1 is None:
        cfg.p1 = cfg.p
    if cfg.p2 is None:
        cfg.p2 = cfg.p
    return cfg
