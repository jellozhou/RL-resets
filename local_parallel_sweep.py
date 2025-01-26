import os
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# function to run one parameter sweep locally
def run_local_sweep(reset_rate, N, learning_rate, gamma, epsilon, n_stable, learning_end_condition, resetting_mode_value, boundary_mode, dimension, reset_decay, num_episodes, render_mode):
    command = [
        "python", "one_sweep_learning_resets.py",
        "--reset_rate", str(reset_rate),
        "--N", str(N),
        "--learning_end_condition", str(learning_end_condition),
        "--learning_rate", str(learning_rate),
        "--gamma", str(gamma),
        "--epsilon", str(epsilon),
        "--n_stable", str(n_stable),
        "--reset_decay", str(reset_decay),
        "--num_episodes", str(num_episodes),
        "--render_mode", str(render_mode),
        "--resetting_mode", str(resetting_mode_value),
        "--boundary", str(boundary_mode), 
        "--dimension", str(dimension)
    ]
    # print("Exec/uting command:", " ".join(command)) # debug
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running: {command}")
        print(e)

def run_local_sweep_with_logging(*args):
    try:
        run_local_sweep(*args)
    except Exception as e:
        print(f"Exception in thread: {e}")

# parameter ranges and parameter list generation
# reset_rates = np.linspace(0.0, 0.005, 101)
# N_array = [25, 30, 35, 40, 45, 50, 55, 60]
N_array = [20]
# reset_rates = [0., 1e-4, 1e-3]
# reset_rates = np.linspace(1e-5, 1e-2, 101)
# reset_rates = np.linspace(0, 1e-6, 11)
# reset_rates = np.linspace(0.,2e-5,101)
reset_rates = [0.005]
learning_rates = [0.0005]
gammas = [0.965]
epsilons = [0.06]
n_stables = [30]
resetting_mode = ["position"]
boundary_mode = ["fixed"] # takes the values periodic or fixed
learning_end_conditions = ["threshold"] # options: threshold, QStable
dimensions = [2]
# threshold: when training is turned off, the agent is able to find the goal in below a threshold
# QStable: the relative magnitudes of the Q-table are stable at the ends of N episodes
# QTable: (implementing) the relative magnitudes of the Q-table satisfy the taxicab

# Generacte parameter list
param_list = []
for dimension in dimensions:
    for learning_end_condition in learning_end_conditions:
        for reset_rate in reset_rates:
            for N in N_array:
                for learning_rate in learning_rates:
                    for gamma in gammas:
                        for epsilon in epsilons:
                            for n_stable in n_stables:
                                for resetting_mode_value in resetting_mode:
                                    for boundary_mode_value in boundary_mode: 
                                        param_list.append((reset_rate, N, learning_rate, gamma, epsilon, n_stable, learning_end_condition, resetting_mode_value, boundary_mode_value, dimension))

# print the number of parameter combinations (to debug)
print(f"Total parameter combinations: {len(param_list)}")

# run the parameter sweep locally
if __name__ == "__main__":
    reset_decay = "twomodes"
    num_episodes = 1 # to save computational power, it would be good to adjust this dynamically...
    render_mode = "None"

    # Parallel execution using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for params in param_list:
            executor.submit(run_local_sweep_with_logging, *params, reset_decay, num_episodes, render_mode)
