import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
import os

# parse variables from filename, return a dict
def extract_variables(filename):
    pattern = r"results/trialnum_(\d+)_resetrate_([\d.]+)_size_(\d+)_dimension_(\d+)_systemsize_(\d+)_learningrate_([\d.]+)_gamma_([\d.]+)_epsilon_([\d.]+)_nstable_(\d+)_learningend_(\w+)_resetdecay_(\w+)_rwd_(\d+)_numepisodes_(\d+)"
    match = re.match(pattern, filename)
    if match:
        return {
            "trial_num": int(match.group(1)),
            "reset_rate": float(match.group(2)),
            "size": int(match.group(3)),
            "dimension": int(match.group(4)),
            "system_size": int(match.group(5)),
            "learning_rate": float(match.group(6)),
            "gamma": float(match.group(7)),
            "epsilon": float(match.group(8)),
            "n_stable": int(match.group(9)),
            "learning_end": match.group(10),
            "reset_decay": match.group(11),
            "reward": int(match.group(12)),
            "num_episodes": int(match.group(13))
        }
    print(f"Warning: Filename '{filename}' did not match the pattern.")
    return {}

# title formatting
def format_title(variables):
    return f"Trial {variables['trial_num']}, Reset Rate {variables['reset_rate']}, Size {variables['size']}, Dimension {variables['dimension']}, System Size {variables['system_size']}, Learning Rate {variables['learning_rate']}, Gamma {variables['gamma']}, Epsilon {variables['epsilon']}, N Stable {variables['n_stable']}, Learning End {variables['learning_end']}, Reset Decay {variables['reset_decay']}, Reward {variables['reward']}, Num Episodes {variables['num_episodes']}"

# Function to determine the base filename by stripping the directory and file extension
def get_base_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

# parse arguments from bash script
parser = argparse.ArgumentParser()
parser.add_argument('--filenames', type=str, nargs='+', default=None)
args = parser.parse_args()
filenames = args.filenames

# Function to extract resetting rate, epsilon, learning rate, and gamma from filename
def extract_resetting_rate_epsilon_learning_rate_gamma(filename):
    variables = extract_variables(filename)
    if 'reset_rate' not in variables:
        print(f"Error: 'reset_rate' not found in variables for filename '{filename}'")
    return variables['reset_rate'], variables['epsilon'], variables['learning_rate'], variables['gamma']

# Function to plot average data with error bars
def plot_avg_with_error_bars(ax, filenames, row_index, ylabel, logscale=False, cumsum=False):
    for filename in filenames:
        if filename.endswith('.csv'):
            filename_stripped = filename[:-4]  # Remove exactly 4 characters from the end

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)
        data_avg = np.loadtxt(filename, delimiter=',')
        std_filename = filename.replace('.csv', '_std.csv')
        data_std = np.loadtxt(std_filename, delimiter=',')

        avg_array = data_avg[row_index]
        std_array = data_std[row_index]  # Assuming standard deviations are stored in the corresponding rows
        episode_num = np.linspace(1, avg_array.shape[0], avg_array.shape[0])

        if cumsum:
            # Compute cumulative sum and cumulative error
            avg_array = np.cumsum(avg_array)
            std_array = np.sqrt(np.cumsum(std_array ** 2))

        # Plot the main data line
        line, = ax.plot(episode_num, avg_array, label=f"{reset_rate}, {epsilon}, {learning_rate}, {gamma}")

        # Plot error bars with the same color as the line
        ax.errorbar(episode_num[::20], avg_array[::20], yerr=std_array[::20], fmt='none', ecolor=line.get_color(), capsize=3)

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

# Function to plot median data
def plot_median(ax, filenames, row_index, ylabel, logscale=False, cumsum=False):
    for filename in filenames:
        if filename.endswith('.csv'):
            filename_stripped = filename[:-4]  # Remove exactly 4 characters from the end

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)
        median_filename = filename.replace(".csv", "_median.csv")
        data_median = np.loadtxt(median_filename, delimiter=',')

        median_array = data_median[row_index]
        if cumsum:
            median_array = np.cumsum(median_array)  # Compute cumulative sum

        episode_num = np.linspace(1, median_array.shape[0], median_array.shape[0])

        ax.plot(episode_num, median_array, label=f"{reset_rate}, {epsilon}, {learning_rate}, {gamma}")

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    # ax.legend(fontsize='small')

# Function to create 1x3 subplots for a single metric
def create_1x3_subplots(filenames, row_index, ylabel, filename_suffix, is_avg=True):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Log-log non-cumsum
    if is_avg:
        plot_avg_with_error_bars(axs[0], filenames, row_index, f"{ylabel}", logscale=True, cumsum=False)
    else:
        plot_median(axs[0], filenames, row_index, f"{ylabel}", logscale=True, cumsum=False)

    # Linear-linear non-cumsum
    if is_avg:
        plot_avg_with_error_bars(axs[1], filenames, row_index, f"{ylabel}", logscale=False, cumsum=False)
    else:
        plot_median(axs[1], filenames, row_index, f"{ylabel}", logscale=False, cumsum=False)

    # Linear-linear cumsum
    if is_avg:
        plot_avg_with_error_bars(axs[2], filenames, row_index, f"{ylabel} (cumulative sum)", logscale=False, cumsum=True)
    else:
        plot_median(axs[2], filenames, row_index, f"{ylabel} (cumulative sum)", logscale=False, cumsum=True)

    # Add a single legend outside the plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    # Save the plot
    base_filename = get_base_filename(filenames[0])
    plt.savefig(f"figs/{base_filename}_{filename_suffix}.png", bbox_inches='tight')
    # plt.show()

# Example usage
create_1x3_subplots(filenames, 2, "Avg Final Path Length", "avg_final_path_length", is_avg=True)
create_1x3_subplots(filenames, 4, "Avg Testing Epilength", "avg_testing_epilength", is_avg=True)
create_1x3_subplots(filenames, 1, "Avg Length", "avg_length", is_avg=True)
create_1x3_subplots(filenames, 2, "Median Final Path Length", "median_final_path_length", is_avg=False)
create_1x3_subplots(filenames, 4, "Median Testing Epilength", "median_testing_epilength", is_avg=False)
create_1x3_subplots(filenames, 1, "Median Length", "median_length", is_avg=False)