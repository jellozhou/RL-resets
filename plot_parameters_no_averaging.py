import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar

# Read data from CSV
filename = "log/parameter_sweep_log_fixed_fpt_test_fewpoints.csv"
# filename = 'log/parameter_sweep_log_fixed_250.csv'
data_fpt = defaultdict(list)  # Data for first passage time
data_regret = defaultdict(list)  # Data for regret
data_learning = defaultdict(list)  # Data for learning episode
data_fcp = defaultdict(list)
N_stable_value = 30  # to filter N_stable values
learning_end_value = 'threshold'

# to do: implement a function that, similar to integrated regret, calculates the sum of total lengths of final paths (not epilengths) across episodes? 


# get theoretical r_opt for 1D and 2D
def objective_function(r, N):
    if r <= 0 or r >= 1:
        return np.inf  # Ensure r is within the valid range (0,1)
    
    term1 = np.exp(np.arccosh(1 / (1 - r))) ** (-N)
    return (1 - term1) / (r * term1)


def derivative_function(r, N):
    if r <= 0 or r >= 1:
        return np.inf  # Avoid invalid values
    
    cosh_term = np.arccosh(1 / (1 - r))
    exp_term = np.exp(-N * cosh_term)
    
    # Compute derivative using the chain rule
    d_cosh = 1 / ((1 - r) * np.sqrt((1 / (1 - r))**2 - 1))  # Derivative of arccosh(1/(1-r))
    d_exp = -N * exp_term * d_cosh  # Derivative of exp(-N * arccosh(1/(1-r)))
    
    numerator = -d_exp * (r * exp_term) - (1 - exp_term) * (exp_term + r * d_exp)
    denominator = (r * exp_term) ** 2
    
    return numerator / denominator

def r_opt_1D_deriv(N):
    result = root_scalar(derivative_function, args=(N,), bracket=(1e-6, 1-1e-6), method='brentq')
    return result.root if result.converged else None

def r_opt_1D(N):
    result = minimize_scalar(objective_function, args=(N,), bounds=(1e-6, 1-1e-6), method='bounded')
    return result.x if result.success else None

def r_opt_2D(r, N): # WRITE THIS!
    return True


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
        # print(first_passage_time)
        first_complete_path = float(row[-2]) # length[0]
        N_stable = int(row[7]) # for new format with number of un-learned trials
        # N_stable = int(row[6]) # OLD FORMAT!
        boundary_type = row[2]
        learning_episode = float(row[-5])
        max_length = int(row[-1])
        ending_regret = float(row[-4])
        print(max_length_across_trials - max_length)
        regret = float(row[-6]) + (max_length_across_trials - max_length) * ending_regret # to adjust for different max lengths

        # only add rows where learning episode != -1.0
        # this happens when there is only one episode (i.e. just to find FPT)
        if learning_episode != 0: # if it's -1, the code now doesn't add it. 
            data_regret[system_size].append((reset_rate, regret))
            data_learning[system_size].append((reset_rate, learning_episode))
            # if reset_rate < 1e-3: # only add fpt below a certain value # not useful anymore
        # if reset_rate < 1e-7: 
        data_fpt[system_size].append((reset_rate, first_passage_time))
        data_fcp[system_size].append((reset_rate, first_complete_path))

# whether or not to calculate sweeping average
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

    # rescaled FPT plot to hopefully see line
for size in sorted(data_fpt.keys()):
    # if size != 30:
    #     continue
    # First Passage Time Plot
    values_fpt = sorted(data_fpt[size])  # Sort by resetting rate
    reset_rates_fpt, fptimes = zip(*values_fpt)
    print(reset_rates_fpt)
    print(fptimes)
    
    min_fptime = min(fptimes)
    min_rate = reset_rates_fpt[fptimes.index(min_fptime)]

    # Rescale FPT
    rescaled_fpt = (np.log(np.array(reset_rates_fpt) * np.array(fptimes) + 1))**2

    # Perform Linear Fit
    slope, intercept, r_value, p_value, std_err = linregress(reset_rates_fpt, rescaled_fpt)
    print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}")

    plt.figure(figsize=(6, 4))
    plt.scatter(reset_rates_fpt, rescaled_fpt, color='blue', label="First Passage Time") 
    plt.plot(reset_rates_fpt, slope * np.array(reset_rates_fpt) + intercept, color='orange', linestyle='--', label=f"Linear Fit (slope={slope:.2e})")
    plt.xlabel("Resetting Rate")
    plt.ylabel("Rescaled FPT")
    # plt.xlim([0, 5e-5])
    # plt.xscale("log")
    # plt.yscale("log")
    plt.title(f"System Size: {size} - Rescaled First Passage Time ({boundary_type}), nstable {N_stable}")
    plt.legend()
    plt.savefig(f"parameter_sweep_figs/size_{size}_rescaled_fpt_{boundary_type}_nstable_{N_stable}_learningend_{learning_end_value}.png")  # Save the figure
    plt.show()

# Normal FPT plot
for size in sorted(data_fpt.keys()):
    r_opt = r_opt_1D_deriv(size)
    print(r_opt)
    # if size != 30:
    #     continue
    # First Passage Time Plot
    values_fpt = sorted(data_fpt[size])  # Sort by resetting rate
    reset_rates_fpt, fptimes = zip(*values_fpt)
    
    min_fptime = min(fptimes)
    min_rate = reset_rates_fpt[fptimes.index(min_fptime)]
    
    plt.figure(figsize=(6, 4))
    plt.scatter(reset_rates_fpt, fptimes, color='blue', label="First Passage Time")
    plt.vlines(r_opt, ymin = min(fptimes), ymax = max(fptimes))
    if calculate_average:
        avg_x, avg_y = calculate_sweeping_average(reset_rates_fpt, fptimes)
        plt.plot(avg_x, avg_y, color='orange', label="Sweeping Average")
    # plt.axvline(min_rate, color='red', linestyle='--', label=f"Optimum {min_rate:.2f}")
    plt.xlabel("Resetting Rate")
    plt.ylabel("First Passage Time")
    # plt.xlim([0, 5e-5])
    # plt.xscale("log")
    # plt.yscale("log")
    plt.title(f"System Size: {size} - First Passage Time ({boundary_type}), nstable {N_stable}")
    plt.legend()
    plt.savefig(f"parameter_sweep_figs/size_{size}_fpt_{boundary_type}_nstable_{N_stable}_learningend_{learning_end_value}.png")  # Save the figure
    plt.show()

# plot for the length of first complete path -- very similar to FPT
# first complete path: FCP
for size in sorted(data_fpt.keys()):
    # if size != 30:
    #     continue
    values_fcp = sorted(data_fcp[size])
    reset_rates_fcp, first_complete_paths = zip(*values_fcp)
    
    min_fcp = min(first_complete_paths)
    min_rate = reset_rates_fcp[first_complete_paths.index(min_fcp)]
    
    plt.figure(figsize=(6, 4))
    plt.scatter(reset_rates_fcp, first_complete_paths, color='blue', label="First Complete Path Length")
    plt.vlines(r_opt, ymin=min(first_complete_paths), ymax=max(first_complete_paths))
    if calculate_average:
        avg_x, avg_y = calculate_sweeping_average(reset_rates_fcp, first_complete_paths)
        plt.plot(avg_x, avg_y, color='orange', label="Sweeping Average")
    # plt.axvline(min_rate, color='red', linestyle='--', label=f"Optimum {min_rate:.2f}")
    plt.xlabel("Resetting Rate")
    plt.ylabel("First Complete Path Length")
    # plt.xlim([0, 5e-5])
    # plt.xscale("log")
    # plt.yscale("log")
    plt.title(f"System Size: {size} - First Complete Path Length ({boundary_type}), nstable {N_stable}")
    plt.legend()
    plt.savefig(f"parameter_sweep_figs/size_{size}_first_complete_path_{boundary_type}_nstable_{N_stable}_learningend_{learning_end_value}.png")  # Save the figure
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
    plt.savefig(f"parameter_sweep_figs/size_{size}_regret_{boundary_type}_nstable_{N_stable}_learningend_{learning_end_value}.png")  # Save the figure
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
    plt.savefig(f"parameter_sweep_figs/size_{size}_learningep_{boundary_type}_nstable_{N_stable}_learningend_{learning_end_value}.png")  # Save the figure
    plt.show()


# # Plot product of first passage time & reset rate
# # this is purely experimental and idt it really works 

# for size in sorted(data_fpt.keys()):
#     # Ensure both fpt and regret data are available
#     if size not in data_regret:
#         print(f"No regret data available for system size {size}")
#         continue

#     # sort and match reset rates for FPT and regret
#     values_fpt = sorted(data_fpt[size])  # Sort by resetting rate
#     values_regret = sorted(data_regret[size])  # Sort by resetting rate

#     reset_rates_fpt, fptimes = zip(*values_fpt)
#     reset_rates_regret, regrets = zip(*values_regret)

#     # match reset rates and calculate product
#     reset_rates = []
#     products = []
#     for rate_fpt, fptime in zip(reset_rates_fpt, fptimes):
#         if rate_fpt in reset_rates_regret:
#             idx = reset_rates_regret.index(rate_fpt)
#             regret = regrets[idx]
#             reset_rates.append(rate_fpt)
#             products.append(fptime * regret)

#     # plotting!
#     plt.figure(figsize=(6, 4))
#     plt.scatter(reset_rates, products, color='red', label="FPT × Regret")
#     if calculate_average:
#         avg_x, avg_y = calculate_sweeping_average(reset_rates, products)
#         plt.plot(avg_x, avg_y, color='orange', label="Sweeping Average")
#     plt.xlabel("Resetting Rate")
#     plt.ylabel("FPT × Regret")
#     plt.title(f"System Size: {size} - Product of FPT and Regret ({boundary_type}), nstable {N_stable}")
#     plt.legend()
#     plt.savefig(f"parameter_sweep_figs/size_{size}_product_fpt_regret_{boundary_type}_nstable_{N_stable}.png")  # Save the figure
#     plt.show()


# plot the optimal resetting rate that minimizes regret, versus system size
min_rates = []
system_sizes = []

for size in sorted(data_regret.keys()):
    values_regret = sorted(data_regret[size])  # sort by resetting rate
    if not values_regret:
        print(f"No regret data available for system size {size}")
        continue

    reset_rates_regret, regrets = zip(*values_regret)

    # smooth data using calculate_sweeping_average, if the flag is true
    if calculate_average:
        avg_x, avg_y = calculate_sweeping_average(reset_rates_regret, regrets)
    else:
        avg_x, avg_y = reset_rates_regret, regrets

    # find the resetting rate that minimizes regret
    min_regret = min(avg_y)
    min_rate = avg_x[avg_y.index(min_regret)]

    min_rates.append(min_rate)
    system_sizes.append(size)

plt.figure(figsize=(8, 6))
plt.plot(system_sizes, min_rates, marker='o', linestyle='-', color='blue', label="optimal resetting rate")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("N")
plt.ylabel("Resetting rate that minimizes regret")
plt.legend()
plt.savefig(f"parameter_sweep_figs/minimal_regret_resetting_rate_nstable_{N_stable}_boundary_{boundary_type}.png")
plt.show()
