def generate_config():
    # Define parameters
    utilities = ['EI_look_ahead_kq', 'EI_look_ahead_mc', 'EI_look_ahead_mlmc']
    kernels = ['matern']
    seeds = [7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    dataset = 'ackley'

    # Open the file to write
    with open(f'/home/zongchen/nest_bq/server_scripts/myriad/{dataset}_configs_1.txt', 'w') as f:
        # Loop over each seed
        for seed in seeds:
            for kernel in kernels:
                for utility in utilities:
                    # Write the command to the file
                    f.write(f"--utility {utility} --dim 2 --N 28 --dataset {dataset} --kernel {kernel} --seed {seed}\n")
            
# Call the function to generate the config file
generate_config()
