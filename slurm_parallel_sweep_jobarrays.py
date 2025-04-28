import os
import numpy as np
import subprocess
import time

# Parameter ranges and parameter list generation
# reset_rates = np.linspace(0, 0.005, 20)
reset_rates = np.linspace(0, 0.0025, 20)
# reset_rates = [reset_rates[20]]
# reset_rates = [1.72413793e-03]
# reset_rates = [0., 0.0005, 0.001, 0.0015, 0.002]
N_array = [20]
# N_array = [20, 30, 50, 100, 200, 500, 1000]  # for FPT sweep
learning_rates = [0.005]
gammas = [0.965]
# epsilons = np.linspace(0.05, 0.95, 20)
epsilons = [0.5]
# epsilons = [0.1]
n_stables = [30]
resetting_mode = ["position"]
reset_decays = ["twomodes"]  # either "none" or "twomodes"
system_sizes = [300, 500] # just enough to have nonzero r^*
learning_end_conditions = ["threshold"]
dimensions = [2]
num_trials = 1
strategies = ["epsilon-greedy"]
rewards = [1]
evaluate_full_training = True # use whenever to measure the final testing episode length every episode
render_mode = "None"
num_episodes = 2000 # hopefully large enough to capture the learning process; if 1: FPT sweep

# row with missing data: reset rate 0.001724137931034483 (just run entire row)
# also run more columnar values (from 0.95 to 1.00)

# Generate parameter list
param_list = []
for dimension in dimensions:
    for learning_end_condition in learning_end_conditions:
        for reset_rate in reset_rates:
            for N in N_array:
                for system_size in system_sizes:
                    for learning_rate in learning_rates:
                        for gamma in gammas:
                            for epsilon in epsilons:
                                for n_stable in n_stables:
                                    for resetting_mode_value in resetting_mode:
                                        for reset_decay in reset_decays:
                                            for strategy in strategies:
                                                for rwd in rewards:
                                                    for trial_num in range(num_trials):
                                                        param_list.append(
                                                            (
                                                                reset_rate,
                                                                N,
                                                                # 3*N, # system size that varies by N (bounded)
                                                                system_size,
                                                                learning_rate,
                                                                gamma,
                                                                epsilon,
                                                                n_stable,
                                                                reset_decay,
                                                                resetting_mode_value,
                                                                learning_end_condition,
                                                                dimension,
                                                                trial_num,
                                                                strategy,
                                                                rwd,
                                                            )
                                                        )

num_jobs = len(param_list)  # Updated array size
print(f"Total parameter combinations: {num_jobs}")

# Generate SLURM job array script as a string
slurm_script = f"""#!/bin/bash
#SBATCH --job-name=rl_sweep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=az9327@princeton.edu
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=all
#SBATCH --array=0-{num_jobs - 1}

module load anaconda3/2024.10

# Get parameters from the list
param_list=({ " ".join([",".join(map(str, params)) for params in param_list]) })
params=${{param_list[$SLURM_ARRAY_TASK_ID]}}

# Convert comma-separated string into individual variables
IFS=',' read -r reset_rate N system_size learning_rate gamma epsilon n_stable reset_decay resetting_mode_value learning_end_condition dimension trial_num strategy rwd <<< "$params"

# Run the Python script with the selected parameters
python one_sweep_learning_resets_forlocal.py --reset_rate $reset_rate \\
                                    --N $N \\
                                    --system_size $system_size \\
                                    --learning_rate $learning_rate \\
                                    --gamma $gamma \\
                                    --epsilon $epsilon \\
                                    --n_stable $n_stable \\
                                    --reset_decay $reset_decay \\
                                    --num_episodes {num_episodes} \\
                                    --render_mode {render_mode} \\
                                    --resetting_mode $resetting_mode_value \\
                                    --learning_end_condition $learning_end_condition \\
                                    --dimension $dimension \\
                                    --trial_num $trial_num \\
                                    --strategy $strategy \\
                                    --rwd $rwd \\
                                    {'--evaluate_full_training' if evaluate_full_training else ''}
"""

# Submit SLURM script directly via stdin
process = subprocess.Popen("sbatch", stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
stdout, _ = process.communicate(slurm_script)

# Extract job ID from sbatch output
job_id = None
for line in stdout.splitlines():
    if "Submitted batch job" in line:
        job_id = line.split()[-1]
        break

if job_id is None:
    raise ValueError("Failed to retrieve SLURM job ID from sbatch output.")

print(f"SLURM job array {job_id} submitted successfully!")

# Wait for SLURM job array to complete
print(f"Waiting for SLURM job array {job_id} to complete...")

while True:
    # Check SLURM job status
    result = subprocess.run(["squeue", "--job", str(job_id)], stdout=subprocess.PIPE, text=True)
    if str(job_id) not in result.stdout:
        print(f"SLURM job array {job_id} has completed.")
        break
    time.sleep(120)  # Wait for 2 minutes before checking again

# Collect output files for plotting
output_files = []
for params in param_list:
    # avg case (old)
    # output_file_avg = f"results/trialnum_{params[11]}_resetrate_{params[0]}_size_{params[1]}_dimension_{params[10]}_systemsize_{params[2]}_learningrate_{params[3]}_gamma_{params[4]}_epsilon_{params[5]}_nstable_{params[6]}_learningend_{params[9]}_resetdecay_{params[7]}_rwd_{params[13]}_numepisodes_{num_episodes}_avg.csv"
    # output_files.append(output_file_avg)
    
    # full case (new)
    output_file_full = f"results/trialnum_{params[11]}_resetrate_{params[0]}_size_{params[1]}_dimension_{params[10]}_systemsize_{params[2]}_learningrate_{params[3]}_gamma_{params[4]}_epsilon_{params[5]}_nstable_{params[6]}_learningend_{params[9]}_resetdecay_{params[7]}_rwd_{params[13]}_numepisodes_{num_episodes}_full_data.npz"
    output_files.append(output_file_full)

print("Output files:", output_files)

# Run the plotting script with all collected output files
cmd = f"python plotting_forlocal.py --filenames {' '.join(output_files)}"
try:
    subprocess.run(cmd, shell=True, check=True)
    print("Plotting completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error while running plotting.py: {e}")