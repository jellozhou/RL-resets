import os
import subprocess
import numpy as np

# Function to generate a SLURM script for a parameter sweep
def generate_slurm_script(param_list, reset_decay, num_episodes, render_mode):
    # Create directories for SLURM scripts and logs if they don't exist
    os.makedirs("slurm_scripts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Write the SLURM script to a file
    filename = "slurm_scripts/rl_sweep.slurm"
    with open(filename, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=rl_sweep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=az9327@princeton.edu
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH --array=0-{len(param_list) - 1}

module load anaconda3/2024.10

# Extract parameters for this job
params=(${param_list[$SLURM_ARRAY_TASK_ID]})

# Run the Python script with the extracted parameters
python one_sweep_learning_resets.py --reset_rate ${{params[0]}} \\
                                --N ${{params[1]}} \\
                                --learning_rate ${{params[2]}} \\
                                --gamma ${{params[3]}} \\
                                --epsilon ${{params[4]}} \\
                                --n_stable ${{params[5]}} \\
                                --reset_decay {reset_decay} \\
                                --num_episodes {num_episodes} \\
                                --render_mode {render_mode} \\
                                --resetting_mode ${{params[6]}} \\
                                --boundary ${{params[7]}} \\
                                --learning_end_condition ${{params[8]}} \\
                                --dimension ${{params[9]}}
""")
    return filename

# Function to submit the SLURM script
def submit_slurm_scripts(param_list, reset_decay, num_episodes, render_mode):
    slurm_script = generate_slurm_script(param_list, reset_decay, num_episodes, render_mode)
    subprocess.run(f"sbatch {slurm_script}", shell=True)  # Submit the SLURM script

# Parameter ranges and parameter list generation
# reset_rates = np.linspace(0.0, 0.005, 101)
reset_rates = np.linspace(1e-5, 0.02, 21)
# N_array = [25, 30, 35, 40, 45, 50, 55, 60]
N_array = [50] # to test
# learning_rates = [0.0]
learning_rates = [0.005]
gammas = [0.965]
epsilons = [0.06]
n_stables = [30]
resetting_mode = ["position"]
boundary = ["fixed"]
learning_end_conditions = ["threshold"]  # threshold or QStable
dimensions = [2]

# Generate parameter list
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
                                        param_list.append(
                                            (
                                                reset_rate,
                                                N,
                                                boundary_mode,
                                                learning_rate,
                                                gamma,
                                                epsilon,
                                                n_stable,
                                                learning_end_condition,
                                                resetting_mode_value,
                                                dimension,
                                            )
                                        )

# Print the number of parameter combinations (to debug)
print(f"Total parameter combinations: {len(param_list)}")

# Run the script and submit while filling in non-swept-over parameters
if __name__ == "__main__":
    submit_slurm_scripts(param_list, reset_decay="twomodes", num_episodes=1000, render_mode="None")  # when learning
    # submit_slurm_scripts(param_list, reset_decay="twomodes", num_episodes=1, render_mode="None")  # just to find first-passage time