import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Read data from CSV
filename = "log/parameter_sweep_log_fixed.csv"  # Replace with your actual file name
data_fpt = defaultdict(list)  # Data for first passage time
data_regret = defaultdict(list)  # Data for regret
data_learning = defaultdict(list)  # Data for learning episode
N_stable_value = 30  # to filter N_stable values

with open(filename, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        reset_rate = float(row[0])
        system_size = int(row[1])
        first_passage_time = float(row[-1])
        N_stable = int(row[6])
        boundary_type = row[2]
        regret = float(row[-4])  # Fourth-to-last column
        learning_episode = float(row[-3])  # Third-to-last column
        data_fpt[system_size].append((reset_rate, first_passage_time))

        # Only add rows where learning episode != -1.0
        if learning_episode != -1.0:
            if N_stable == N_stable_value:
                data_regret[system_size].append((reset_rate, regret))
                data_learning[system_size].append((reset_rate, learning_episode))

# Toggle for calculating and displaying sweeping average
calculate_average = True

def calculate_sweeping_average(x, y, window_size=20):
    x, y = np.array(x), np.array(y)
    sorted_indices = np.argsort(x)
    x, y = x[sorted_indices], y[sorted_indices]
    averaged_x = []
    averaged_y = []
    for i in range(len(x) - window_size + 1):
        averaged_x.append(np.mean(x[i:i + window_size]))
        averaged_y.append(np.mean(y[i:i + window_size]))
    return averaged_x, averaged_y

# Generate plots for each system size
for size in sorted(data_fpt.keys()):
    # if size != 30:
    #     continue
    # First Passage Time Plot
    values_fpt = sorted(data_fpt[size])  # Sort by resetting rate
    reset_rates_fpt, fptimes = zip(*values_fpt)
    
    min_fptime = min(fptimes)
    min_rate = reset_rates_fpt[fptimes.index(min_fptime)]
    
    plt.figure(figsize=(6, 4))
    plt.scatter(reset_rates_fpt, fptimes, color='blue', label="First Passage Time")
    if calculate_average:
        avg_x, avg_y = calculate_sweeping_average(reset_rates_fpt, fptimes)
        plt.plot(avg_x, avg_y, color='orange', label="Sweeping Average")
    # plt.axvline(min_rate, color='red', linestyle='--', label=f"Optimum {min_rate:.2f}")
    plt.xlabel("Resetting Rate")
    plt.ylabel("First Passage Time")
    # plt.xscale("log")
    # plt.yscale("log")
    plt.title(f"System Size: {size} - First Passage Time ({boundary_type}), nstable {N_stable}")
    plt.legend()
    # plt.grid(True)
    plt.savefig(f"parameter_sweep_figs/size_{size}_fpt_{boundary_type}_nstable_{N_stable}.png")  # Save the figure
    plt.show()

    # Regret Plot
    values_regret = sorted(data_regret[size])  # Sort by resetting rate
    if not values_regret:
        print(f"No regret data available for system size {size}")
        continue
    reset_rates_regret, regrets = zip(*values_regret)
    
    plt.figure(figsize=(6, 4))
    plt.scatter(reset_rates_regret, regrets, color='green', label="Regret")
    if calculate_average:
        avg_x, avg_y = calculate_sweeping_average(reset_rates_regret, regrets)
        plt.plot(avg_x, avg_y, color='orange', label="Sweeping Average")
    plt.xlabel("Resetting Rate")
    plt.ylabel("Regret")
    # plt.xscale("log")
    # plt.yscale("log")
    plt.title(f"System Size: {size} - Regret ({boundary_type}), nstable {N_stable}")
    plt.legend()
    # plt.grid(True)
    plt.savefig(f"parameter_sweep_figs/size_{size}_regret_{boundary_type}_nstable_{N_stable}.png")  # Save the figure
    plt.show()

    # Learning Episode Plot
    values_learning = sorted(data_learning[size])  # Sort by resetting rate
    reset_rates_learning, learning_episodes = zip(*values_learning)
    
    plt.figure(figsize=(6, 4))
    plt.scatter(reset_rates_learning, learning_episodes, color='purple', label="Learning Episode")
    if calculate_average:
        avg_x, avg_y = calculate_sweeping_average(reset_rates_learning, learning_episodes)
        plt.plot(avg_x, avg_y, color='orange', label="Sweeping Average")
    plt.xlabel("Resetting Rate")
    plt.ylabel("Learning Episode")
    plt.title(f"System Size: {size} - Learning Episode ({boundary_type}), nstable {N_stable}")
    plt.legend()
    # plt.grid(True)
    plt.savefig(f"parameter_sweep_figs/size_{size}_learningep_{boundary_type}_nstable_{N_stable}.png")  # Save the figure
    plt.show()


# Plot Product of First Passage Time and Regret vs Reset Rates
for size in sorted(data_fpt.keys()):
    # Ensure both fpt and regret data are available
    if size not in data_regret:
        print(f"No regret data available for system size {size}")
        continue

    # Sort and match reset rates for FPT and regret
    values_fpt = sorted(data_fpt[size])  # Sort by resetting rate
    values_regret = sorted(data_regret[size])  # Sort by resetting rate

    reset_rates_fpt, fptimes = zip(*values_fpt)
    reset_rates_regret, regrets = zip(*values_regret)

    # Match reset rates and calculate product
    reset_rates = []
    products = []
    for rate_fpt, fptime in zip(reset_rates_fpt, fptimes):
        if rate_fpt in reset_rates_regret:
            idx = reset_rates_regret.index(rate_fpt)
            regret = regrets[idx]
            reset_rates.append(rate_fpt)
            products.append(fptime * regret)

    # Plot the product
    plt.figure(figsize=(6, 4))
    plt.scatter(reset_rates, products, color='red', label="FPT × Regret")
    if calculate_average:
        avg_x, avg_y = calculate_sweeping_average(reset_rates, products)
        plt.plot(avg_x, avg_y, color='orange', label="Sweeping Average")
    plt.xlabel("Resetting Rate")
    plt.ylabel("FPT × Regret")
    plt.title(f"System Size: {size} - Product of FPT and Regret ({boundary_type}), nstable {N_stable}")
    plt.legend()
    plt.savefig(f"parameter_sweep_figs/size_{size}_product_fpt_regret_{boundary_type}_nstable_{N_stable}.png")  # Save the figure
    plt.show()