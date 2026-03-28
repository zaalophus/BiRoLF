from main import run_main, run_movieLens
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
    #                     trials=5,
    #                     horizon=3000,
    #                     arm_x=20,
    #                     arm_y=20,
    #                     true_dim_x=10,
    #                     true_dim_y=10,
    #                     dim_x=7,
    #                     dim_y=7,
    #                     case=2,
    #                     explore=True,
    #                     init_explore="double",
    #                     timing_breakdown=True,
    #                     profile_ops=True,
    #                     timing_log_every=50,
    #                     sequential_benchmark=True,
    #                     seed = 5421,
    #                     kappa_cap_percentile = 0.0)
    # run_main(now_script.cfg)
    
    # now_script = Script(cfg,
    #                     trials=5,
    #                     horizon=3000,
    #                     arm_x=25,
    #                     arm_y=25,
    #                     true_dim_x=20,
    #                     true_dim_y=20,
    #                     dim_x=10,
    #                     dim_y=10,
    #                     case=4,
    #                     explore=True,
    #                     init_explore="double",
    #                     timing_breakdown=True,
    #                     profile_ops=True,
    #                     timing_log_every=50,
    #                     sequential_benchmark=True,
    #                     seed = 1212,
    #                     kappa_cap_percentile = 0.0)
    # run_main(now_script.cfg)
    
    # now_script = Script(cfg,
    #                     trials=5,
    #                     horizon=3000,
    #                     arm_x=25,
    #                     arm_y=25,
    #                     true_dim_x=15,
    #                     true_dim_y=15,
    #                     dim_x=10,
    #                     dim_y=10,
    #                     case=5,
    #                     explore=True,
    #                     init_explore="double",
    #                     timing_breakdown=True,
    #                     profile_ops=True,
    #                     timing_log_every=50,
    #                     sequential_benchmark=True,
    #                     seed = 1234,
    #                     kappa_cap_percentile = 0.0)
    # run_main(now_script.cfg)
    
# default case : true_dim_x − arm_x < dim_x < arm_x -> arm_x=10,  true_dim_x=14, dim_x=5 
    for seed in [555,21234,624,651,154,726,943]:
        for true_dim in [15,20,25]:
            for num_arm in [20,30,25]:
                for dim in [7, 10,12]:
                    for case in [1,2,4,5]:
                        # if true_dim == 15 and num_arm == 20 and dim ==7 and case in [1,2,4,5]:
                        #     now_script = Script(cfg,
                        #         trials=5,
                        #         horizon=4000,
                        #         arm_x=num_arm,
                        #         arm_y=num_arm,
                        #         true_dim_x=true_dim,
                        #         true_dim_y=true_dim,
                        #         dim_x=dim,
                        #         dim_y=dim,
                        #         case=case,
                        #         explore=True,
                        #         init_explore="half",
                        #         timing_breakdown=True,
                        #         profile_ops=True,
                        #         timing_log_every=50,
                        #         sequential_benchmark=True,
                        #         seed = seed,
                        #         kappa_cap_percentile = 0)
                        #     run_main(now_script.cfg)
                            
                        now_script = Script(cfg,
                            trials=5,
                            horizon=9000,
                            arm_x=num_arm,
                            arm_y=num_arm,
                            true_dim_x=true_dim,
                            true_dim_y=true_dim,
                            dim_x=dim,
                            dim_y=dim,
                            case=case,
                            explore=True,
                            init_explore="half",
                            timing_breakdown=True,
                            profile_ops=True,
                            timing_log_every=50,
                            sequential_benchmark=True,
                            seed = seed,
                            kappa_cap_percentile = 0)
                        # run_main(now_script.cfg)
                        run_movieLens(now_script.cfg,sampling=True,n_sample=25)
                        
                        
                        
                        # elif true_dim ==20 and num_arm ==25 and dim ==10:
                        #     now_script = Script(cfg,
                        #         trials=5,
                        #         horizon=12500,
                        #         arm_x=num_arm,
                        #         arm_y=num_arm,
                        #         true_dim_x=true_dim,
                        #         true_dim_y=true_dim,
                        #         dim_x=dim,
                        #         dim_y=dim,
                        #         case=case,
                        #         explore=True,
                        #         init_explore="K",
                        #         timing_breakdown=True,
                        #         profile_ops=True,
                        #         timing_log_every=50,
                        #         sequential_benchmark=True,
                        #         seed = seed,
                        #         kappa_cap_percentile = 0)
                        #     run_main(now_script.cfg)
                            
                        # if true_dim - num_arm < dim and dim < num_arm and true_dim > dim:
                        #     now_script = Script(cfg,
                        #         trials=5,
                        #         horizon=2000,
                        #         arm_x=num_arm,
                        #         arm_y=num_arm,
                        #         true_dim_x=true_dim,
                        #         true_dim_y=true_dim,
                        #         dim_x=dim,
                        #         dim_y=dim,
                        #         case=case,
                        #         explore=True,
                        #         init_explore="double",
                        #         timing_breakdown=True,
                        #         profile_ops=True,
                        #         timing_log_every=50,
                        #         sequential_benchmark=True,
                        #         seed = seed,
                        #         kappa_cap_percentile = 0)
                        #     run_main(now_script.cfg)

