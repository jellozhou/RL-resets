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
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH --array=0-{len(param_list) - 1}

module load anaconda3/2024.10

param_list=(
""")
        for params in param_list:
            f.write(f"\"{params[0]} {params[1]} {params[2]} {params[3]} {params[4]} {params[5]} {params[6]} {params[7]} {params[8]} {params[9]} {params[10]}\"\n")
        f.write(")\n")

        f.write("""
params=(${param_list[$SLURM_ARRAY_TASK_ID]})
reset_rate=${params[0]}
N=${params[1]}
system_size=${params[2]}
learning_rate=${params[3]}
gamma=${params[4]}
epsilon=${params[5]}
n_stable=${params[6]}
learning_end_condition=${params[7]}
resetting_mode=${params[8]}
dimension=${params[9]}
trial_num=${params[10]}

python one_sweep_learning_resets.py --reset_rate $reset_rate \\
                                    --N $N \\
                                    --system_size $system_size \\
                                    --learning_rate $learning_rate \\
                                    --gamma $gamma \\
                                    --epsilon $epsilon \\
                                    --n_stable $n_stable \\
                                    --reset_decay {reset_decay} \\
                                    --num_episodes {num_episodes} \\
                                    --render_mode {render_mode} \\
                                    --resetting_mode $resetting_mode \\
                                    --learning_end_condition $learning_end_condition \\
                                    --dimension $dimension \\
                                    --trial_num $trial_num
""")
    return filename

# Function to submit the SLURM script
def submit_slurm_scripts(param_list, reset_decay, num_episodes, render_mode):
    slurm_script = generate_slurm_script(param_list, reset_decay, num_episodes, render_mode)
    subprocess.run(f"sbatch {slurm_script}", shell=True)  # Submit the SLURM script

# Parameter ranges and parameter list generation
reset_rates = np.linspace(1e-5, 0.02, 11)
N_array = [20]  # to test
learning_rates = [0.005]
gammas = [0.965]
epsilons = [0.06]
n_stables = [30]
resetting_mode = ["position"]
system_sizes = [3 * N for N in N_array]  # replace this with the fixed boundary condition
learning_end_conditions = ["threshold"]  # threshold or QStable
dimensions = [2]
num_trials = 1  # Number of trials per parameter set

# Generate parameter list
param_list = []
for dimension in dimensions:
    for learning_end_condition in learning_end_conditions:
        for system_size in system_sizes:
            for reset_rate in reset_rates:
                for N in N_array:
                    for learning_rate in learning_rates:
                        for gamma in gammas:
                            for epsilon in epsilons:
                                for n_stable in n_stables:
                                    for resetting_mode_value in resetting_mode:
                                        for trial_num in range(num_trials):
                                            param_list.append(
                                                (
                                                    reset_rate,
                                                    N,
                                                    system_size,
                                                    learning_rate,
                                                    gamma,
                                                    epsilon,
                                                    n_stable,
                                                    learning_end_condition,
                                                    resetting_mode_value,
                                                    dimension,
                                                    trial_num,
                                                )
                                            )

# Print the number of parameter combinations (to debug)
print(f"Total parameter combinations: {len(param_list)}")

# Run the script and submit while filling in non-swept-over parameters
if __name__ == "__main__":
    submit_slurm_scripts(param_list, reset_decay="twomodes", num_episodes=1000, render_mode="None")  # when learning
    # submit_slurm_scripts(param_list, reset_decay="twomodes", num_episodes=1, render_mode="None")  # just to find first-passage time
