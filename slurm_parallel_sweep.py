import os
import subprocess
import numpy as np

# perform (hopefully faster) parameter sweep by generating many slurm scripts and running all of them, in which each script calls one_sweep_learning_resets.py (the sweep for one parameter value)

# function to generate a SLURM script for each parameter combination
def generate_slurm_script(reset_rate, N, learning_rate, gamma, epsilon, n_stable, learning_end_condition, resetting_mode_value, boundary_mode, reset_decay, num_episodes, render_mode):
    script_content = f"""#!/bin/bash
#SBATCH --job-name=sweep_rr{reset_rate:.4f}_N_{N}_lr{learning_rate}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=az9327@princeton.edu
#SBATCH --output=logs/sweep_rr{reset_rate:.4f}_N_{N}_lr{learning_rate}.out
#SBATCH --error=logs/sweep_rr{reset_rate:.4f}_N_{N}_lr{learning_rate}.err
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G

module load anaconda3/2024.10

python one_sweep_learning_resets.py --reset_rate {reset_rate} \\
                                --N {N} \\
                                --learning_rate {learning_rate} \\
                                --gamma {gamma} \\
                                --epsilon {epsilon} \\
                                --n_stable {n_stable} \\
                                --reset_decay {reset_decay} \\
                                --num_episodes {num_episodes} \\
                                --render_mode {render_mode} \\
                                --resetting_mode {resetting_mode_value} \\
                                --boundary {boundary_mode} \\
                                --learning_end_condition {learning_end_condition}
"""
    # Create directories for SLURM scripts and logs if they don't exist
    os.makedirs("slurm_scripts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Write the script to a file
    filename = f"slurm_scripts/sweep_rr{reset_rate:.4f}_lr{learning_rate}_nstable{n_stable}.slurm"
    with open(filename, "w") as f:
        f.write(script_content)
    return filename

# function to submit all generated scripts to the SLURM scheduler
def submit_slurm_scripts(param_list, reset_decay, num_episodes, render_mode):
    for params in param_list:
        reset_rate, N, boundary_mode, learning_rate, gamma, epsilon, n_stable, learning_end_condition, resetting_mode_value = params
        slurm_script = generate_slurm_script(reset_rate, N, learning_rate, gamma, epsilon, n_stable, learning_end_condition, resetting_mode_value, boundary_mode, reset_decay, num_episodes, render_mode)
        subprocess.run(f"sbatch {slurm_script}", shell=True) # run the slurm script

# parameter ranges and parameter list generation
reset_rates = np.linspace(0.0,0.005,101)
N_array = [25, 30, 35, 40, 45, 50, 55, 60]
# learning_rates = [0.0005]
learning_rates = [0.]
gammas = [0.965]
epsilons = [0.06]
n_stables = [30]
resetting_mode = ["position"]
boundary = ["fixed"]
learning_end_conditions = ["threshold"] # threshold or QStable

# generate parameter list
param_list = []
for learning_end_condition in learning_end_conditions:
    for boundary_mode in boundary:
        for reset_rate in reset_rates:
            for N in N_array:
                for learning_rate in learning_rates:
                    for gamma in gammas:
                        for epsilon in epsilons:
                            for n_stable in n_stables:
                                for resetting_mode_value in resetting_mode: 
                                    param_list.append((reset_rate, N, boundary_mode, learning_rate, gamma, epsilon, n_stable, learning_end_condition, resetting_mode_value))

# print the number of parameter combinations (to debug)
print(f"Total parameter combinations: {len(param_list)}")

# run the script t and submit while filling in non-swept-over parameters
if __name__ == "__main__":
    submit_slurm_scripts(param_list, reset_decay="twomodes", num_episodes=400, render_mode="None") # when learning
    # submit_slurm_scripts(param_list, reset_decay="twomodes", num_episodes=1, render_mode="None") # just to find first-passage time