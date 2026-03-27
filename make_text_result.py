import pickle

if __name__ == "__main__":
    base_path = "/home/jihyeongpark/repo/BiRoLF/4. Rebuttal/"
    
    target_file_path = "exp_half_seed_41415_arm_30_dim_12_true_dim_25/results/2026-03-26/case_1_seed_41415_p_0.6_std_0.1/"
    target_file = "Case_1_M_30_N_30_xstar_25_ystar_25_dx_12_dy_12_T_9000_explored_half_noise_0.1_run_1748.pkl"
    
    with open(base_path + target_file_path + target_file, "rb") as f:
        data = pickle.load(f)
        for i in data[1]:
            print(i)
            
        print(len(data[1]["RoLF"][0]))