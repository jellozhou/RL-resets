import os
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, timeout=600)  # Add a timeout of 600 seconds
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}")
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {cmd}")
    except KeyboardInterrupt:
        print("Process interrupted by user")
        raise

# Collect all output files for plotting
output_files = []

def run_sweep(param_list, num_episodes, render_mode, evaluate_full_training):
    commands = [
        f"python one_sweep_learning_resets.py --reset_rate {params[0]} --N {params[1]} --system_size {params[2]} --learning_rate {params[3]} --gamma {params[4]} --epsilon {params[5]} --n_stable {params[6]} --reset_decay {params[7]} --num_episodes {num_episodes} --render_mode {render_mode} --resetting_mode {params[8]} --learning_end_condition {params[9]} --dimension {params[10]} --trial_num {params[11]} --strategy {params[12]} --rwd {params[13]} {'--evaluate_full_training' if evaluate_full_training else ''}"
        for params in param_list
    ]
    
    for cmd in commands: # debug
        print(cmd)  # Print each command for debugging
    
    with ThreadPoolExecutor(max_workers=3) as executor:  # Limit the number of parallel processes (3 for now)
        try:
            executor.map(run_command, commands)
        except KeyboardInterrupt:
            print("Parallel execution interrupted by user")
            executor.shutdown(wait=False)
            raise

    # Collect output files for plotting
    for params in param_list:
        output_file_avg = f"results/trialnum_{params[11]}_resetrate_{params[0]}_size_{params[1]}_dimension_{params[10]}_systemsize_{params[2]}_learningrate_{params[3]}_gamma_{params[4]}_epsilon_{params[5]}_nstable_{params[6]}_learningend_{params[9]}_resetdecay_{params[7]}_rwd_{params[13]}_numepisodes_{num_episodes}.csv"
        output_files.append(output_file_avg)

# Parameter ranges and parameter list generation
reset_rates = np.linspace(0, 0.002, 5)
# reset_rates = [0.0005] # to test
N_array = [20]  # to test
learning_rates = [0.005]
gammas = [0.965]
epsilons = [0.875]
n_stables = [30]
resetting_mode = ["position"]
reset_decays = ["none"] # either "none" or "twomodes": "none" used to generate testing data, "twomodes" for big regret sweeps
system_sizes = [3 * N for N in N_array]  # replace this with the fixed boundary condition
# system_sizes = [int(200)] # for the unbounded case
learning_end_conditions = ["threshold"]  # threshold or QStable
dimensions = [2]
num_trials = 1  # Number of trials per parameter set
strategies = ["epsilon-greedy"] # options: epsilon-greedy or softmax
rewards = [1]  # Example reward values for reaching the goal

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
                                        for reset_decay in reset_decays:
                                            for strategy in strategies:
                                                for rwd in rewards:
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
                                                                reset_decay,
                                                                resetting_mode_value,
                                                                learning_end_condition,
                                                                dimension,
                                                                trial_num,
                                                                strategy,
                                                                rwd,
                                                            )
                                                        )

# Print the parameter list for debugging
for params in param_list:
    print(params)

# Print the number of parameter combinations (to debug)
print(f"Total parameter combinations: {len(param_list)}")

# Run the script and submit while filling in non-swept-over parameters
if __name__ == "__main__":
    evaluate_full_training = True  # Set this flag when seeking full info of the testing phase in every episode; i.e. never stop evaluating testing epi length when training technically done
    run_sweep(param_list, num_episodes=200, render_mode="None", evaluate_full_training=evaluate_full_training)  # when learning
    # Call the plotting script with all collected output files
    cmd = f"python plotting.py --filenames {' '.join(output_files)}"
    print(output_files)
    try:
        run_command(cmd)
    except RuntimeError as e:
        print(e)
