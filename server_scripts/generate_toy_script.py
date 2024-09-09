# Define the command template
command_template = "python simulated.py --seed {seed} --kernel_x matern --kernel_theta matern"

# Open the file toy_local.sh in write mode
with open('/home/zongchen/nest_bq/configs/toy_local.sh', 'w') as file:
    # Write a shebang line to make the script executable on Unix systems
    file.write("#!/bin/bash\n")
    
    # Generate commands for seeds 0 to 100 and write each to the file
    for seed in range(101):
        command = command_template.format(seed=seed)
        file.write(command + '\n')

# Make the script executable (works on Unix-based systems)
import os
os.chmod('/home/zongchen/nest_bq/configs/toy_local.sh', 0o755)

print("Commands written to toy_local.sh and the script is now executable.")