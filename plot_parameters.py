import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize_scalar, root_scalar

# Read data from CSV
filename = "log/parameter_sweep_log_fixed_fpt_test_fewpoints.csv"
data_fpt = defaultdict(list)  # Data for first passage time
data_regret = defaultdict(list)  # Data for regret
data_learning = defaultdict(list)  # Data for learning episode
data_fcp = defaultdict(list)
N_stable_value = 30  # to filter N_stable values
learning_end_value = 'threshold'

# Create dictionaries for averaged data
averaged_fpt = defaultdict(list)
averaged_regret = defaultdict(list)
averaged_learning = defaultdict(list)
averaged_fcp = defaultdict(list)

# Calculate averages over all trials for each reset rate and system size
def average_data(data_dict):
    averaged_data = defaultdict(list)
    for size, values in data_dict.items():
        summed_values = defaultdict(lambda: [0, 0])  # [sum, count] for each (reset_rate)
        for rate, value in values:
            summed_values[rate][0] += value  # Sum of values
            summed_values[rate][1] += 1  # Count of entries
        for rate, (total, count) in summed_values.items():
            averaged_data[size].append((rate, total / count))  # Calculate the average
    return averaged_data

# Read data from the CSV
with open(filename, 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

    # first loop through rows to find maximum r_opt episode length across trials
    max_length_across_trials = 0
    for row in rows:
        max_length = int(row[-1])
        if max_length > max_length_across_trials:
            max_length_across_trials = max_length

    for row in rows:
        reset_rate = float(row[0])
        system_size = int(row[1])
        first_passage_time = float(row[-3])
        first_complete_path = float(row[-2])  # length[0]
        N_stable = int(row[7])  # for new format with number of un-learned trials
        boundary_type = row[2]
        learning_episode = float(row[-5])
        max_length = int(row[-1])
        ending_regret = float(row[-4])
        regret = float(row[-6]) + (max_length_across_trials - max_length) * ending_regret  # Adjust for different max lengths

        # only add rows where learning episode != -1.0
        if learning_episode != 0:
            data_regret[system_size].append((reset_rate, regret))
            data_learning[system_size].append((reset_rate, learning_episode))

        data_fpt[system_size].append((reset_rate, first_passage_time))
        data_fcp[system_size].append((reset_rate, first_complete_path))

# Calculate averages for all data
averaged_fpt = average_data(data_fpt)
averaged_regret = average_data(data_regret)
averaged_learning = average_data(data_learning)
averaged_fcp = average_data(data_fcp)

# Plot for each system size with averaged data
for size in sorted(averaged_fpt.keys()):
    # First Passage Time Plot
    values_fpt = sorted(averaged_fpt[size])  # Sort by resetting rate
    reset_rates_fpt, fptimes = zip(*values_fpt)

    plt.figure(figsize=(6, 4))
    plt.scatter(reset_rates_fpt, fptimes, color='blue', label="First Passage Time")
    plt.xlabel("Resetting Rate")
    plt.ylabel("First Passage Time")
    plt.title(f"System Size: {size} - First Passage Time ({boundary_type}), nstable {N_stable_value}")
    plt.legend()
    plt.savefig(f"parameter_sweep_figs/size_{size}_fpt_averaged_{boundary_type}_nstable_{N_stable_value}_learningend_{learning_end_value}.png")
    plt.show()

    # Regret Plot
    values_regret = sorted(averaged_regret[size])  # Sort by resetting rate
    reset_rates_regret, regrets = zip(*values_regret)

    plt.figure(figsize=(6, 4))
    plt.scatter(reset_rates_regret, regrets, color='green', label="Regret")
    plt.xlabel("Resetting Rate")
    plt.ylabel("Regret")
    plt.title(f"System Size: {size} - Regret ({boundary_type}), nstable {N_stable_value}")
    plt.legend()
    plt.savefig(f"parameter_sweep_figs/size_{size}_regret_averaged_{boundary_type}_nstable_{N_stable_value}_learningend_{learning_end_value}.png")
    plt.show()

    # Learning Episode Plot
    values_learning = sorted(averaged_learning[size])  # Sort by resetting rate
    reset_rates_learning, learning_episodes = zip(*values_learning)

    plt.figure(figsize=(6, 4))
    plt.scatter(reset_rates_learning, learning_episodes, color='purple', label="Learning Episode")
    plt.xlabel("Resetting Rate")
    plt.ylabel("Learning Episode")
    plt.title(f"System Size: {size} - Learning Episode ({boundary_type}), nstable {N_stable_value}")
    plt.legend()
    plt.savefig(f"parameter_sweep_figs/size_{size}_learningep_averaged_{boundary_type}_nstable_{N_stable_value}_learningend_{learning_end_value}.png")
    plt.show()

    # First Complete Path Length Plot
    values_fcp = sorted(averaged_fcp[size])
    reset_rates_fcp, first_complete_paths = zip(*values_fcp)

    plt.figure(figsize=(6, 4))
    plt.scatter(reset_rates_fcp, first_complete_paths, color='blue', label="First Complete Path Length")
    plt.xlabel("Resetting Rate")
    plt.ylabel("First Complete Path Length")
    plt.title(f"System Size: {size} - First Complete Path Length ({boundary_type}), nstable {N_stable_value}")
    plt.legend()
    plt.savefig(f"parameter_sweep_figs/size_{size}_first_complete_path_averaged_{boundary_type}_nstable_{N_stable_value}_learningend_{learning_end_value}.png")
    plt.show()

# If needed, add more plots for other data types.
