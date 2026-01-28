from main import run_main
from cfg import get_cfg

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
            kappa_cap_percentile = 0.99):
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
    # now_script = Script(cfg,
    #     trials=5,
    #     horizon=2000,
    #     arm_x=20,
    #     arm_y=20,
    #     true_dim_x=20,
    #     true_dim_y=20,
    #     dim_x=5,
    #     dim_y=5,
    #     case=2,
    #     explore=False,
    #     init_explore="double",
    #     timing_breakdown=True,
    #     profile_ops=True,
    #     timing_log_every=50,
    #     sequential_benchmark=True,
    #     seed = 2)
    # run_main(now_script.cfg)
    
# default case : true_dim_x − arm_x < dim_x < arm_x -> arm_x=10,  true_dim_x=14, dim_x=5 
    for seed in [2121,4343]:
        for true_dim in [20,15]:
            for num_arm in [18,25]:
                for dim in [10]:
                    for case in [1,2,4,5]:
                        if true_dim - num_arm < dim and dim < num_arm and true_dim > dim:
                            now_script = Script(cfg,
                                trials=5,
                                horizon=2000,
                                arm_x=num_arm,
                                arm_y=num_arm,
                                true_dim_x=true_dim,
                                true_dim_y=true_dim,
                                dim_x=dim,
                                dim_y=dim,
                                case=case,
                                explore=True,
                                init_explore="double",
                                timing_breakdown=True,
                                profile_ops=True,
                                timing_log_every=50,
                                sequential_benchmark=True,
                                seed = seed,
                                kappa_cap_percentile = 0.99)
                            run_main(now_script.cfg)

