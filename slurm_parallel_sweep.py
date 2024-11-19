import os
import subprocess
import numpy as np

# perform (hopefully faster) parameter sweep by generating many slurm scripts and running all of them, in which each script calls one_sweep_learning_resets.py (the sweep for one parameter value)

# function to generate a SLURM script for each parameter combination
def generate_slurm_script(reset_rate, learning_rate, gamma, epsilon, n_stable, qlearn_after_resets_value, resetting_mode_value, reset_decay, num_episodes, render_mode):
    script_content = f"""#!/bin/bash
#SBATCH --job-name=sweep_rr{reset_rate:.4f}_lr{learning_rate}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=az9327@princeton.edu
#SBATCH --output=logs/sweep_rr{reset_rate:.4f}_lr{learning_rate}.out
#SBATCH --error=logs/sweep_rr{reset_rate:.4f}_lr{learning_rate}.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

module load anaconda3/2024.2

python one_sweep_learning_resets.py --reset_rate {reset_rate} \\
                                --learning_rate {learning_rate} \\
                                --gamma {gamma} \\
                                --epsilon {epsilon} \\
                                --n_stable {n_stable} \\
                                --reset_decay {reset_decay} \\
                                --num_episodes {num_episodes} \\
                                --render_mode {render_mode} \\
                                --qlearn_after_resets {qlearn_after_resets_value} \\
                                --resetting_mode {resetting_mode_value}
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
        reset_rate, learning_rate, gamma, epsilon, n_stable, qlearn_after_resets_value, resetting_mode_value = params
        slurm_script = generate_slurm_script(reset_rate, learning_rate, gamma, epsilon, n_stable, qlearn_after_resets_value, resetting_mode_value, reset_decay, num_episodes, render_mode)
        subprocess.run(f"sbatch {slurm_script}", shell=True) # run the slurm script

# parameter ranges and parameter list generation
reset_rates = np.linspace(0.05,0.1,501)
learning_rates = [0.0005]
gammas = [0.965]
epsilons = [0.06]
n_stables = [30]
qlearn_after_resets = [True]
resetting_mode = ["position"]

# generate parameter list
param_list = []
for reset_rate in reset_rates:
    for learning_rate in learning_rates:
        for gamma in gammas:
            for epsilon in epsilons:
                for n_stable in n_stables:
                    for qlearn_after_resets_value in qlearn_after_resets:
                        for resetting_mode_value in resetting_mode: 
                            param_list.append((reset_rate, learning_rate, gamma, epsilon, n_stable, qlearn_after_resets_value, resetting_mode_value))

# print the number of parameter combinations (to debug)
print(f"Total parameter combinations: {len(param_list)}")

# run the script generator and submit while filling in non-swept-over parameters
if __name__ == "__main__":
    submit_slurm_scripts(param_list, reset_decay="twomodes", num_episodes=300, render_mode="None")