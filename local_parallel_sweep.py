import os
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# function to run one parameter sweep locally
def run_local_sweep(reset_rate, N, learning_rate, gamma, epsilon, n_stable, qlearn_after_resets_value, resetting_mode_value, boundary_mode, reset_decay, num_episodes, render_mode):
    command = [
        "python", "one_sweep_learning_resets.py",
        "--reset_rate", str(reset_rate),
        "--N", str(N),
        "--learning_rate", str(learning_rate),
        "--gamma", str(gamma),
        "--epsilon", str(epsilon),
        "--n_stable", str(n_stable),
        "--reset_decay", reset_decay,
        "--num_episodes", str(num_episodes),
        "--render_mode", render_mode,
        "--qlearn_after_resets", str(qlearn_after_resets_value),
        "--resetting_mode", resetting_mode_value,
        "--boundary", boundary_mode
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running: {command}")
        print(e)

# parameter ranges and parameter list generation
# reset_rates = np.linspace(0.0, 0.005, 101)
# N_array = [25, 30, 35, 40, 45, 50, 55, 60]
N_array = [50]
# reset_rates = np.linspace(0.,2e-5,101)
reset_rates = [0.002]
learning_rates = [0.0005]
gammas = [0.965]
epsilons = [0.06]
n_stables = [5] # to test
qlearn_after_resets = [False]
resetting_mode = ["position"]
boundary_mode = ["fixed"] # takes the values periodic or fixed

# Generacte parameter list
param_list = []
for reset_rate in reset_rates:
    for N in N_array:
        for learning_rate in learning_rates:
            for gamma in gammas:
                for epsilon in epsilons:
                    for n_stable in n_stables:
                        for qlearn_after_resets_value in qlearn_after_resets:
                            for resetting_mode_value in resetting_mode:
                                for boundary_mode_value in boundary_mode: 
                                    param_list.append((reset_rate, N, learning_rate, gamma, epsilon, n_stable, qlearn_after_resets_value, resetting_mode_value, boundary_mode_value))

# print the number of parameter combinations (to debug)
print(f"Total parameter combinations: {len(param_list)}")

# run the parameter sweep locally
if __name__ == "__main__":
    reset_decay = "twomodes"
    num_episodes = 10
    render_mode = "None"

    # Parallel execution using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for params in param_list:
            executor.submit(run_local_sweep, *params, reset_decay, num_episodes, render_mode)