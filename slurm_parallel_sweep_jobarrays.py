import os
import numpy as np
import subprocess

# Parameter ranges and parameter list generation
#reset_rates = np.linspace(0.0, 0.005, 101)
reset_rates = np.linspace(1e-5, 0.015, 21)
N_array = [50]
learning_rates = [0.005]
gammas = [0.965]
epsilons = [0.06]
n_stables = [30]
resetting_mode = ["position"]
boundary = ["fixed"]
learning_end_conditions = ["threshold"]  # threshold or QStable
dimensions = [2]
num_trials = 10  # Number of trials per parameter set

# Generate expanded parameter list (each parameter set is repeated num_trials times)
param_list = []
for dimension in dimensions:
    for learning_end_condition in learning_end_conditions:
        for boundary_mode in boundary:
            for reset_rate in reset_rates:
                for N in N_array:
                    for learning_rate in learning_rates:
                        for gamma in gammas:
                            for epsilon in epsilons:
                                for n_stable in n_stables:
                                    for resetting_mode_value in resetting_mode:
                                        for trial_num in range(num_trials):
                                            param_list.append((
                                                reset_rate, N, boundary_mode, learning_rate, gamma, 
                                                epsilon, n_stable, learning_end_condition, 
                                                resetting_mode_value, dimension, trial_num
                                            ))

num_jobs = len(param_list)  # Updated array size
print(f"Total parameter combinations: {num_jobs}")

# Generate SLURM job array script as a string
slurm_script = f"""#!/bin/bash
#SBATCH --job-name=rl_sweep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=az9327@princeton.edu
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=all
#SBATCH --array=0-{num_jobs - 1}

module load anaconda3/2024.10

# Get parameters from the list
param_list=({ " ".join([",".join(map(str, params)) for params in param_list]) })
params=${{param_list[$SLURM_ARRAY_TASK_ID]}}

# Convert comma-separated string into individual variables
IFS=',' read -r reset_rate N boundary_mode learning_rate gamma epsilon n_stable learning_end_condition resetting_mode_value dimension trial_num <<< "$params"

# Run the Python script with the selected parameters
python one_sweep_learning_resets.py --reset_rate $reset_rate \\
                                    --N $N \\
                                    --learning_rate $learning_rate \\
                                    --gamma $gamma \\
                                    --epsilon $epsilon \\
                                    --n_stable $n_stable \\
                                    --reset_decay twomodes \\
                                    --num_episodes 1000 \\
                                    --render_mode None \\
                                    --resetting_mode $resetting_mode_value \\
                                    --boundary $boundary_mode \\
                                    --learning_end_condition $learning_end_condition \\
                                    --dimension $dimension \\
                                    --trial_num $trial_num
"""

# Submit SLURM script directly via stdin
process = subprocess.Popen("sbatch", stdin=subprocess.PIPE, text=True)
process.communicate(slurm_script)

print("SLURM job array submitted successfully!")