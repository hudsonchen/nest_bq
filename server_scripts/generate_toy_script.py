# Define the command template
command_template = "--seed {seed} --kernel_x matern --kernel_theta matern --N_T_ratio {ratio} --d {d} --scale {scale}"

# Open the file toy_local.sh in write mode
with open('/home/zongchen/nest_bq/server_scripts/myriad/toy_configs_1.txt', 'w') as file:
    # Generate commands for seeds 0 to 100 and write each to the file
    for seed in range(101):
        for ratio in [1.0]:
            for d in [1]:
                for scale in [1.0]:
                    command = command_template.format(seed=seed, ratio=ratio, d=d, scale=scale)
                    file.write(command + '\n')

# Make the script executable (works on Unix-based systems)
import os
os.chmod('/home/zongchen/nest_bq/server_scripts/myriad/toy_configs_1.txt', 0o755)
