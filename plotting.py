import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

# packages to loop over a directory and generate plots
import os
import fnmatch

# parse variables from filename, return a dict
def extract_variables(filename):
    # print(filename)
    parts = filename.split('_')  # Split the filename by underscores
    variables = {}
    # loop through the parts two at a time (variable and value)
    for i in range(0, len(parts) - 1, 2):
        variable = parts[i]
        value = parts[i + 1]
        # convert variables to numeric values
        if re.match(r'^-?\d+(\.\d+)?$', value):  # Check if the value is a number (integer or float)
            value = float(value) if '.' in value else int(value)
        # exclude 'numepisodes' from being added to the dictionary cuz it's redundant
        if variable.lower() != "numepisodes":
            variables[variable] = value  # Add the variable and its value to the dictionary
    return variables

# title formatting
def format_title(variables):
    # create a string representation and join them with newlines for better formatting
    title_str = ", ".join([f"{key}: {value}" for key, value in variables.items()])
    return title_str

# parse arguments from bash script
parser = argparse.ArgumentParser()
parser.add_argument('--filenames', type=str, nargs='+', default=None)
args = parser.parse_args()
filenames = args.filenames

# Function to extract resetting rate from filename
def extract_resetting_rate(filename):
    variables = extract_variables(filename)
    return variables.get('resetrate', 'unknown')

# Function to plot data from multiple files
def plot_data(ax, filenames, row_index, ylabel, logscale=False):
    for filename in filenames:
        if filename.endswith('.csv'):
            filename_stripped = filename[:-4]  # Remove exactly 4 characters from the end

        resetting_rate = extract_resetting_rate(filename_stripped)
        avg_array = np.loadtxt(filename, delimiter=',')
        episode_num = np.linspace(1, avg_array.shape[1], avg_array.shape[1])

        ax.plot(episode_num, avg_array[row_index], label=f"{resetting_rate}")  # Plot the specified row

    ax.set_title("Comparison of Different Resetting Rates", fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel)
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.legend(fontsize='small')

# Function to plot cumulative sum data from multiple files
def plot_cumsum_data(ax, filenames, row_index, ylabel, logscale=False):
    for filename in filenames:
        if filename.endswith('.csv'):
            filename_stripped = filename[:-4]  # Remove exactly 4 characters from the end

        resetting_rate = extract_resetting_rate(filename_stripped)
        avg_array = np.loadtxt(filename, delimiter=',')
        episode_num = np.linspace(1, avg_array.shape[1], avg_array.shape[1])
        cumsum_data = np.cumsum(avg_array[row_index])

        ax.plot(episode_num, cumsum_data, label=f"{resetting_rate}")  # Plot the cumulative sum

    ax.set_title("Cumulative Sum Comparison of Different Resetting Rates", fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel)
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.legend(fontsize='small')

def smooth_data(data, window_size=10):
    """Smooth the data using a simple moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Function to plot data from multiple files with smoothing
def plot_smoothed_data(ax, filenames, row_index, ylabel, logscale=False, window_size=25):
    for filename in filenames:
        if filename.endswith('.csv'):
            filename_stripped = filename[:-4]  # Remove exactly 4 characters from the end

        resetting_rate = extract_resetting_rate(filename_stripped)
        avg_array = np.loadtxt(filename, delimiter=',')
        episode_num = np.linspace(1, avg_array.shape[1], avg_array.shape[1])
        smoothed_data = smooth_data(avg_array[row_index], window_size)

        ax.plot(episode_num[:len(smoothed_data)], smoothed_data, label=f"{resetting_rate}")  # Plot the smoothed data

    ax.set_title("Smoothed Comparison of Different Resetting Rates", fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel)
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.legend(fontsize='small')

# Function to plot cumulative sum data from multiple files with smoothing
def plot_smoothed_cumsum_data(ax, filenames, row_index, ylabel, logscale=False, window_size=25):
    for filename in filenames:
        if filename.endswith('.csv'):
            filename_stripped = filename[:-4]  # Remove exactly 4 characters from the end

        resetting_rate = extract_resetting_rate(filename_stripped)
        avg_array = np.loadtxt(filename, delimiter=',')
        episode_num = np.linspace(1, avg_array.shape[1], avg_array.shape[1])
        smoothed_data = smooth_data(avg_array[row_index], window_size)
        cumsum_smoothed_data = np.cumsum(smoothed_data)

        ax.plot(episode_num[:len(cumsum_smoothed_data)], cumsum_smoothed_data, label=f"{resetting_rate}")  # Plot the cumulative sum of smoothed data

    ax.set_title("Cumulative Sum Smoothed Comparison of Different Resetting Rates", fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel)
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.legend(fontsize='small')

# Create subplots for each evaluation type
def create_subplots(filenames, row_index, ylabel, filename_prefix):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.tight_layout(pad=5.0)

    plot_smoothed_data(axs[0], filenames, row_index, f"Smoothed {ylabel}")
    plot_smoothed_cumsum_data(axs[1], filenames, row_index, f"Smoothed cumulative sum of {ylabel}")
    plot_smoothed_data(axs[2], filenames, row_index, f"Smoothed {ylabel}", logscale=True)
    plot_smoothed_cumsum_data(axs[3], filenames, row_index, f"Smoothed cumulative sum of {ylabel}", logscale=True)

    plt.savefig(f"figs/{filename_prefix}_plots.png", bbox_inches='tight')
    # plt.show()

# Plot average reward per episode (row 0)
create_subplots(filenames, 0, "average reward per episode", "avg_reward")

# Plot average length per episode (row 1)
create_subplots(filenames, 1, "average length per episode", "avg_epilength")

# Plot average length of final path per episode (row 2)
create_subplots(filenames, 2, "average length of final path per episode", "avg_length")

# Plot average regret per episode (row 3)
create_subplots(filenames, 3, "average regret per episode", "avg_regret")

# Plot average testing episode length per episode (row 4)
create_subplots(filenames, 4, "average testing episode length per episode", "avg_testing_epilength")