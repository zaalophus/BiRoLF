from main import run_main, run_movieLens
from cfg import get_cfg
import subprocess

class Script:
    def __init__(self, cfg,
                 trials=1,
            horizon=2000,
            arm_x=20,
            arm_y=20,
            true_dim_x=15,
            true_dim_y=15,
            dim_x=7,
            dim_y=7,
            case=1,
            explore=True,
            init_explore="double",
            timing_breakdown=True,
            profile_ops=True,
            timing_log_every=50,
            sequential_benchmark=True,
            seed = 1,
            kappa_cap_percentile = 0.99,
            d_unobs_movie = 4,
            d_unobs_user = 4,
            use_embedding = True,
            use_random_user_obs = True,
            n_random_user_obs = 2,
            block_oo_max_iter = 30,
            block_ou_max_iter = 20,
            block_uo_max_iter = 20,
            block_tol = 1e-4):
        self.cfg = cfg
        self.cfg.trials = trials
        self.cfg.horizon = horizon
        self.cfg.arm_x = arm_x
        self.cfg.arm_y = arm_y
        self.cfg.true_dim_x = true_dim_x
        self.cfg.true_dim_y = true_dim_y
        self.cfg.dim_x = dim_x
        self.cfg.dim_y = dim_y
        self.cfg.case = case
        self.cfg.explore = explore
        self.cfg.init_explore = init_explore
        self.cfg.timing_breakdown = timing_breakdown
        self.cfg.profile_ops = profile_ops
        self.cfg.timing_log_every = timing_log_every
        self.cfg.sequential_benchmark = sequential_benchmark
        self.cfg.seed = seed
        self.cfg.kappa_cap_percentile = kappa_cap_percentile

        self.cfg.d_unobs_movie = d_unobs_movie
        self.cfg.d_unobs_user  = d_unobs_user
        self.cfg.use_embedding = use_embedding
        self.cfg.use_random_user_obs = use_random_user_obs
        self.cfg.n_random_user_obs = n_random_user_obs
        
        self.cfg.block_oo_max_iter = block_oo_max_iter
        self.cfg.block_ou_max_iter = block_ou_max_iter
        self.cfg.block_uo_max_iter = block_uo_max_iter
        self.cfg.block_tol = block_tol

# python main.py 
# —trials 1 
# —horizon 2000
# —arm_x 20 
# —arm_y 20 
# —true_dim_x 15 
# —true_dim_y 15 
# —dim_x 7 
# —dim_y 7 
# —case 1 
# —explore True 
# —init_explore double 
# —timing_breakdown True 
# —profile_ops True 
# —timing_log_every 50 
# —sequential_benchmark True
     
if __name__ == "__main__":
    cfg = get_cfg()
# default case : true_dim_x − arm_x < dim_x < arm_x -> arm_x=10,  true_dim_x=14, dim_x=5 

    d_unobs_movie = 10
    d_unobs_user = 10
    use_embedding = True
    use_random_user_obs = True
    n_random_user_obs = 2
    
    session_name = subprocess.check_output(['tmux','display-message','-p','#S'],text=True).strip()
    print(session_name)
    
    # base_seed = 30000
    # interval = 20
    # for seed in range(base_seed+interval*int(session_name[-1])-(interval-1),base_seed+int(session_name[-1])*interval):
    #     for case in [1,2,4,5]:
    #         now_script = Script(cfg,
    #             trials=5,
    #             horizon=4000,
    #             case=case,
    #             explore=True,
    #             init_explore="half",
    #             timing_breakdown=True,
    #             profile_ops=True,
    #             timing_log_every=50,
    #             sequential_benchmark=False,
    #             seed = seed,
    #             kappa_cap_percentile = 0,
    #             d_unobs_movie = d_unobs_movie,
    #             d_unobs_user = d_unobs_user,
    #             block_oo_max_iter = 200,
    #             block_ou_max_iter = 200,
    #             block_uo_max_iter = 200,
    #             block_tol=1e-4,
    #             use_embedding = use_embedding,
    #             use_random_user_obs=use_random_user_obs,
    #             n_random_user_obs=n_random_user_obs)
    #         run_main(now_script.cfg)
                        
    start_seed = 50000
    interval = 20
    end_seed = start_seed + interval
    for seed in range(start_seed,end_seed):
        for case in [1,2,4,5]:
            now_script = Script(cfg,
                trials=5,
                horizon=4000,
                case=case,
                explore=True,
                init_explore="half",
                timing_breakdown=True,
                profile_ops=True,
                timing_log_every=50,
                sequential_benchmark=False,
                seed = seed,
                kappa_cap_percentile = 0,
                d_unobs_movie = d_unobs_movie,
                d_unobs_user = d_unobs_user,
                block_oo_max_iter = 200,
                block_ou_max_iter = 200,
                block_uo_max_iter = 200,
                block_tol=1e-4,
                use_embedding = use_embedding,
                use_random_user_obs=use_random_user_obs,
                n_random_user_obs=n_random_user_obs)
            run_movieLens(now_script.cfg,sampling=True,n_sample=25)
    
                        
    # list_seed = [2111,2112,2113,2114,2115,2116,2117,2118,2119,2120]
    # list_true_dim = [15,20,25]
    # list_num_arm = [20,30,25]
    # list_dim = [7, 10,12]
    # list_horizon = [4000,9000,12500]
    # list_case = [1,2,4,5]
    
    # for seed in list_seed:
    #     for case in list_case:
    #         for true_dim, num_arm, dim, hor in zip(list_true_dim, list_num_arm, list_dim, list_horizon):
    #             now_script = Script(cfg,
    #                         trials=5,
    #                         horizon=hor,
    #                         arm_x=num_arm,
    #                         arm_y=num_arm,
    #                         true_dim_x=true_dim,
    #                         true_dim_y=true_dim,
    #                         dim_x=dim,
    #                         dim_y=dim,
    #                         case=case,
    #                         explore=True,
    #                         init_explore="half",
    #                         timing_breakdown=True,
    #                         profile_ops=True,
    #                         timing_log_every=50,
    #                         sequential_benchmark=True,
    #                         seed = seed,
    #                         kappa_cap_percentile = 0)
    #             run_main(now_script.cfg)
    
    