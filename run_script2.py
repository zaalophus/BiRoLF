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
            seed = 1):
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
       
# default case : true_dim_x − arm_x < dim_x < arm_x -> arm_x=10,  true_dim_x=14, dim_x=5 
if __name__ == "__main__":
    cfg = get_cfg()
    for seed in [1,2,3,4]:
        for case in range(1,10):
            for init_explore in ["double"]:
                now_script = Script(cfg,
                    trials=5,
                    horizon=2000,
                    arm_x=20,
                    arm_y=20,
                    true_dim_x=15,
                    true_dim_y=15,
                    dim_x=7,
                    dim_y=7,
                    case=case,
                    explore=True,
                    init_explore=init_explore,
                    timing_breakdown=True,
                    profile_ops=True,
                    timing_log_every=50,
                    sequential_benchmark=True,
                    seed = seed)
                run_main(now_script.cfg)