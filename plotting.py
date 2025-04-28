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
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    # print("base filename:", os.path.splitext(os.path.basename(filepath))[0])

    # Remove only the specific fields from the filename while keeping other parts intact, except resetdecay
    pattern = r"(trialnum_\d+|resetrate_[\d.]+|dimension_\d+|learningrate_[\d.]+|gamma_[\d.]+|nstable_\d+|learningend_\w+|rwd_\d+|numepisodes_\d+)"
    stripped_filename = re.sub(pattern, "", base_filename)
    resetdecay_match = re.search(r"resetdecay_\w+", base_filename)
    if resetdecay_match:
        resetdecay_match = re.match(r"resetdecay_[a-zA-Z]+", resetdecay_match.group(0))  # Match only "resetdecay_" followed by letters
    if resetdecay_match:
        stripped_filename = resetdecay_match.group(0) + "_" + stripped_filename
    stripped_filename = re.sub(r"__+", "_", stripped_filename).strip("_")  # Clean up extra underscores
    # print("stripped filename:", stripped_filename)  # debug
    return stripped_filename

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
        std_filename = filename.replace('_avg.csv', '_std.csv')
        data_std = np.loadtxt(std_filename, delimiter=',')

        avg_array = data_avg[row_index]
        # print("avg_array", avg_array)  # debug
        std_array = data_std[row_index]  # Assuming standard deviations are stored in the corresponding rows
        episode_num = np.linspace(1, avg_array.shape[0], avg_array.shape[0])

        if cumsum:
            # Compute cumulative sum and cumulative error
            avg_array = np.cumsum(avg_array)
            std_array = np.sqrt(np.cumsum(std_array ** 2))

        # Determine indices to plot 10 error bars evenly spaced
        error_bar_indices = np.linspace(0, len(episode_num) - 1, 10, dtype=int)

        # Plot the main data line
        line, = ax.plot(episode_num, avg_array, label=f"{reset_rate}, {epsilon}, {learning_rate}, {gamma}")

        # Plot error bars with the same color as the line
        ax.errorbar(episode_num[error_bar_indices], avg_array[error_bar_indices], yerr=std_array[error_bar_indices], 
                    fmt='none', ecolor=line.get_color(), capsize=3)

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

# Function to plot median (not mean!!) of the medians
def plot_median(ax, filenames, row_index, ylabel, logscale=False, cumsum=False):
    for filename in filenames:
        if filename.endswith('_avg.csv'):
            filename_stripped = filename[:-4]  # Remove exactly 4 characters from the end

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)
        median_filename = filename.replace("_avg.csv", "_median.csv")
        data_median = np.loadtxt(median_filename, delimiter=',')
        # print("median data", data_median) # debug

        # median_array = np.median(data_median, axis=0)  # Compute the median of the medians
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

# Function to plot smoothed max of the max
def plot_smoothed_max(ax, filenames, row_index, ylabel, logscale=False, cumsum=False, smoothing_window=10):
    for filename in filenames:
        if filename.endswith('.csv'):
            filename_stripped = filename[:-4]  # Remove exactly 4 characters from the end

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)
        max_filename = filename.replace("_avg.csv", "_max.csv")  # assume this file exists
        data_max = np.loadtxt(max_filename, delimiter=',')

        max_array = data_max[row_index]  # Assuming max values are stored in the corresponding rows
        if cumsum:
            max_array = np.cumsum(max_array)  # Compute cumulative sum

        # Apply smoothing using a simple moving average
        smoothed_max_array = np.convolve(max_array, np.ones(smoothing_window) / smoothing_window, mode='valid')

        episode_num = np.linspace(1, len(smoothed_max_array), len(smoothed_max_array))

        ax.plot(episode_num, smoothed_max_array, label=f"{reset_rate}, {epsilon}, {learning_rate}, {gamma}")

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

# Function to plot smoothed min of the min
def plot_smoothed_min(ax, filenames, row_index, ylabel, logscale=False, cumsum=False, smoothing_window=10):
    for filename in filenames:
        if filename.endswith('.csv'):
            filename_stripped = filename[:-4]  # Remove exactly 4 characters from the end

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)
        min_filename = filename.replace("_avg.csv", "_min.csv")  # assume this file exists
        data_min = np.loadtxt(min_filename, delimiter=',')

        min_array = data_min[row_index]  # Assuming min values are stored in the corresponding rows
        if cumsum:
            min_array = np.cumsum(min_array)  # Compute cumulative sum

        # Apply smoothing using a simple moving average
        smoothed_min_array = np.convolve(min_array, np.ones(smoothing_window) / smoothing_window, mode='valid')

        episode_num = np.linspace(1, len(smoothed_min_array), len(smoothed_min_array))

        ax.plot(episode_num, smoothed_min_array, label=f"{reset_rate}, {epsilon}, {learning_rate}, {gamma}")

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

# Function to create 1x3 subplots for a single metric
def create_1x3_subplots(filenames, row_index, ylabel, filename_prefix, type):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # different plotting options depending on the type of data we look at
    if type == 'mean':
        plot_avg_with_error_bars(axs[0], filenames, row_index, f"{ylabel}", logscale=True, cumsum=False)
        plot_avg_with_error_bars(axs[1], filenames, row_index, f"{ylabel}", logscale=False, cumsum=False)
        plot_avg_with_error_bars(axs[2], filenames, row_index, f"{ylabel} (cumulative sum)", logscale=False, cumsum=True)
    elif type == 'median':
        plot_median(axs[0], filenames, row_index, f"{ylabel}", logscale=True, cumsum=False)
        plot_median(axs[1], filenames, row_index, f"{ylabel}", logscale=False, cumsum=False)
        plot_median(axs[2], filenames, row_index, f"{ylabel} (cumulative sum)", logscale=False, cumsum=True)
    elif type == 'max':
        plot_smoothed_max(axs[0], filenames, row_index, f"{ylabel}", logscale=True, cumsum=False)
        plot_smoothed_max(axs[1], filenames, row_index, f"{ylabel}", logscale=False, cumsum=False)
        plot_smoothed_max(axs[2], filenames, row_index, f"{ylabel} (cumulative sum)", logscale=False, cumsum=True)
    elif type == 'min':
        plot_smoothed_min(axs[0], filenames, row_index, f"{ylabel}", logscale=True, cumsum=False)
        plot_smoothed_min(axs[1], filenames, row_index, f"{ylabel}", logscale=False, cumsum=False)
        plot_smoothed_min(axs[2], filenames, row_index, f"{ylabel} (cumulative sum)", logscale=False, cumsum=True)

    # Add a single legend outside the plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    # Save the plot
    base_filename = get_base_filename(filenames[0])
    plt.savefig(f"figs/{filename_prefix}_{base_filename}.png", bbox_inches='tight')
    # plt.show()

# call plotting function for each metric
create_1x3_subplots(filenames, 1, "Avg Final Path Length", "avg_final_path_length", type='mean')
create_1x3_subplots(filenames, 3, "Avg Testing Epilength", "avg_testing_epilength", type='mean')
create_1x3_subplots(filenames, 0, "Avg Length", "avg_length", type='mean')

create_1x3_subplots(filenames, 1, "Median Final Path Length", "median_final_path_length", type='median')
create_1x3_subplots(filenames, 3, "Median Testing Epilength", "median_testing_epilength", type='median')
create_1x3_subplots(filenames, 0, "Median Length", "median_length", type='median')

create_1x3_subplots(filenames, 1, "Maximum Final Path Length (smoothed)", "max_final_path_length", type='max')
create_1x3_subplots(filenames, 3, "Maximum Testing Epilength (smoothed)", "max_testing_epilength", type='max')
create_1x3_subplots(filenames, 0, "Maximum Length (smoothed)", "max_length", type='max')

create_1x3_subplots(filenames, 1, "Minimum Final Path Length (smoothed)", "min_final_path_length", type='min')
create_1x3_subplots(filenames, 3, "Minimum Testing Epilength (smoothed)", "min_testing_epilength", type='min')
create_1x3_subplots(filenames, 0, "Minimum Length (smoothed)", "min_length", type='min')

# example filename
# trialnum_0_resetrate_0.0015_size_20_dimension_2_systemsize_60_learningrate_0.005_gamma_0.965_epsilon_0.875_nstable_30_learningend_threshold_resetdecay_none_rwd_1_numepisodes_200_avg.csv