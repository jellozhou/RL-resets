import os
import numpy as np
import itertools

# Define parameter ranges
reset_rates = np.linspace(0, 0.01, 101)
N_array = [25, 30, 35, 40, 45, 50, 55, 60]
learning_rates = [0.0005]
gammas = [0.965]
epsilons = [0.06]
n_stables = [30]
learning_end_conditions = ["threshold"]
resetting_mode = ["position"]
boundary_modes = ["fixed"]
dimensions = [2]
reset_decay = ["twomodes"]
num_episodes = 400
render_mode = "none"

# Create results directories
os.makedirs("results", exist_ok=True)
os.makedirs("figs", exist_ok=True)
os.makedirs("vectors", exist_ok=True)

# Function to count available trials
def count_trials(reset_rate, N, learning_rate, gamma, epsilon, n_stable, learning_end_condition, boundary_mode, dimension, reset_decay, resetting_mode):
    trial = 1
    while True:
        output_file = f"results/resetrate_{reset_rate}_size_{N}_dimension_{dimension}_boundary_{boundary_mode}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_numepisodes_{num_episodes}_trial_{trial}.csv"
        if not os.path.exists(output_file):
            return trial - 1  # Last successful trial count
        trial += 1

# Function to compute averages across trials and save to log file
def compute_averages(reset_rate, N, learning_rate, gamma, epsilon, n_stable, learning_end_condition, boundary_mode, dimension, reset_decay, resetting_mode):
    num_trials = count_trials(reset_rate, N, learning_rate, gamma, epsilon, n_stable, learning_end_condition, boundary_mode, dimension, reset_decay, resetting_mode)
    
    if num_trials == 0:
        print(f"Skipping {reset_rate}, {N}, {learning_rate}, {gamma}, {epsilon}, {n_stable}, {learning_end_condition} (No trials found)")
        return
    
    print(f"Processing {reset_rate}, {N}, {learning_rate}, {gamma}, {epsilon}, {n_stable}, {learning_end_condition} with {num_trials} trials")

    max_length = 1
    reward_sum = np.zeros(max_length)
    epilength_sum = np.zeros(max_length)
    regret_sum = np.zeros(max_length)
    length_sum = np.zeros(max_length)

    training_done_epi_sum = 0
    ending_regret_sum = 0
    num_training_not_done = 0

    for trial in range(1, num_trials + 1):
        vectors = {}
        for vec_name in ["total_reward_vec", "total_epilength_vec", "total_regret_vec", "total_length_vec", "training_done_epi", "ending_regret_file"]:
            vec_file = f"vectors/{vec_name}_resetrate_{reset_rate}_size_{N}_dimension_{dimension}_boundary_{boundary_mode}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
            if not os.path.exists(vec_file):
                print(f"  Skipping trial {trial}, missing {vec_name}")
                return
            vectors[vec_name] = np.load(vec_file, allow_pickle=True)

        max_length = max(max_length, len(vectors["total_reward_vec"]))
        reward_sum = np.pad(reward_sum, (0, max_length - len(reward_sum)), mode='constant')
        epilength_sum = np.pad(epilength_sum, (0, max_length - len(epilength_sum)), mode='constant')
        regret_sum = np.pad(regret_sum, (0, max_length - len(regret_sum)), mode='constant')
        length_sum = np.pad(length_sum, (0, max_length - len(length_sum)), mode='constant')

        reward_sum += np.pad(vectors["total_reward_vec"], (0, max_length - len(vectors["total_reward_vec"])), mode='edge')
        epilength_sum += np.pad(vectors["total_epilength_vec"], (0, max_length - len(vectors["total_epilength_vec"])), mode='edge')
        regret_sum += np.pad(vectors["total_regret_vec"], (0, max_length - len(vectors["total_regret_vec"])), mode='edge')
        length_sum += np.pad(vectors["total_length_vec"], (0, max_length - len(vectors["total_length_vec"])), mode='edge')

        if vectors["training_done_epi"] != -1:
            training_done_epi_sum += vectors["training_done_epi"]
        else: # if training is not done
            num_training_not_done += 1

        ending_regret_sum += vectors["ending_regret_file"]

    reward_avg = reward_sum / num_trials
    epilength_avg = epilength_sum / num_trials
    regret_avg = regret_sum / num_trials
    length_avg = length_sum / num_trials
    training_done_epi_avg = training_done_epi_sum / num_trials
    ending_regret_avg = ending_regret_sum / num_trials

    total_regret_across_episodes = np.sum(regret_avg)
    total_finallength_across_episodes = np.sum(length_avg)

    output_file_avg = f"results/resetrate_{reset_rate}_size_{N}_dimension_{dimension}_boundary_{boundary_mode}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_numepisodes_{num_episodes}.csv"
    np.savetxt(output_file_avg, [reward_avg, epilength_avg, regret_avg], delimiter=",")
    
    log_file = "log/averaging_log.csv"
    with open(log_file, 'a') as f:              
        f.write(f"{reset_rate},{N},{boundary_mode},{num_training_not_done},{learning_rate},{gamma},{epsilon},{n_stable},{reset_decay},{total_regret_across_episodes},{total_finallength_across_episodes},{training_done_epi_avg},{ending_regret_avg},{epilength_avg[0]},{length_avg[0]},{max_length}\n") # there is the indexing since it's a vector
    print(f"Averages written into log file")

# Iterate over all parameter combinations
for params in itertools.product(reset_rates, N_array, learning_rates, gammas, epsilons, n_stables, learning_end_conditions, boundary_modes, dimensions, reset_decay, resetting_mode):
    compute_averages(*params)