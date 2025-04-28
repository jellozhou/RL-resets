import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
import os
import itertools
import argparse
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from itertools import combinations
from collections import Counter
from matplotlib.cm import get_cmap
from scipy.stats import skew
from scipy.ndimage import gaussian_filter # to smooth heatmaps

# exponential fit later
def exponential(x, A, tau):
    return A * np.exp(-x / tau)

# parser = argparse.ArgumentParser()
# parser.add_argument('--filelist', type=str, help="Path to a file containing a list of filenames")

# # Parse arguments from command-line
# args = parser.parse_args()

# # Read filenames from --filelist
# with open(args.filelist, "r") as f:
#     filenames = [line.strip() for line in f.readlines()]

# parse variables from filename, return a dict
def extract_variables(filename):
    pattern = r"results/trialnum_(\d+)_resetrate_([\d.eE+-]+)_size_(\d+)_dimension_(\d+)_systemsize_(\d+)_learningrate_([\d.eE+-]+)_gamma_([\d.eE+-]+)_epsilon_([\d.eE+-]+)_nstable_(\d+)_learningend_(\w+)_resetdecay_(\w+)_rwd_(\d+)_numepisodes_(\d+)"
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
    pattern = r"(trialnum_\d+|resetrate_[\d.eE+-]+|dimension_\d+|learningrate_[\d.eE+-]+|gamma_[\d.eE+-]+|nstable_\d+|learningend_\w+|rwd_\d+|numepisodes_\d+)"
    stripped_filename = re.sub(pattern, "", base_filename)
    resetdecay_match = re.search(r"resetdecay_\w+", base_filename)
    if resetdecay_match:
        resetdecay_match = re.match(r"resetdecay_[a-zA-Z]+", resetdecay_match.group(0))  # Match only "resetdecay_" followed by letters
    if resetdecay_match:
        stripped_filename = resetdecay_match.group(0) + "_" + stripped_filename
    stripped_filename = re.sub(r"__+", "_", stripped_filename).strip("_")  # Clean up extra underscores
    # print("stripped filename:", stripped_filename)  # debug
    return stripped_filename


parser = argparse.ArgumentParser(description="Plot results from output files.")
parser.add_argument("--filenames", nargs="+", required=True, help="Output files to process")
args = parser.parse_args()

print(args.filenames)

# Read filenames from --filenames
if args.filenames:
    filenames = args.filenames  # Directly use the list of filenames
else:
    filenames = None

# print(filenames)

# some error script that clogs things up:
# if filenames:
#     print('filenames:', filenames)
# else:
#     print("No filenames provided. Please pass filenames using the --filenames argument.")

# Function to extract resetting rate, epsilon, learning rate, and gamma from filename
def extract_resetting_rate_epsilon_learning_rate_gamma(filename):
    variables = extract_variables(filename)
    if 'reset_rate' not in variables:
        print(f"Error: 'reset_rate' not found in variables for filename '{filename}'")
    return variables['reset_rate'], variables['epsilon'], variables['learning_rate'], variables['gamma']

# Function to extract epilength, length, and testing_epilength from filenames
def extract_full_data_from_filenames(filenames):
    epilength_all_trials = {}
    length_all_trials = {}
    testing_epilength_all_trials = {}
    training_done_epi_sum = {} # only computes the sum of the episodes that have successfully converged on a good solution
    
    for filename in filenames:
        try:
            if filename.endswith('.npz'):
                data = np.load(filename)
                epilength_all_trials[filename] = data["epilength_all_trials"]
                length_all_trials[filename] = data["length_all_trials"]
                testing_epilength_all_trials[filename] = data["testing_epilength_all_trials"]
                training_done_epi_sum[filename] = data["training_done_epi_sum"]
        except FileNotFoundError:
            print(f"Warning: File {filename} not found, skipping...")
            continue
        except Exception as e:
            print(f"Warning: Error loading {filename}: {str(e)}, skipping...")
            continue

        print(training_done_epi_sum)

    if not epilength_all_trials:
        print("Warning: No valid data files found")
        return {}, {}, {}

    return epilength_all_trials, length_all_trials, testing_epilength_all_trials, training_done_epi_sum


# Define a color cycle with the first 10 colors from 'tab10' and the next 10 from 'tab20b'
# this is to keep stylistic consistency 
color_cycle = list(plt.cm.get_cmap('tab10', 10).colors) + list(plt.cm.get_cmap('tab20b', 10).colors)

# extract data from filenames to then use
epilength_all_trials, length_all_trials, testing_epilength_all_trials, training_done_epi_sum = extract_full_data_from_filenames(filenames)

def find_intersection_point(data_dict):
    # Parameters
    window_length = 21
    polyorder = 3
    cutoff_frac = 0.05
    bin_width = 2 # seems to work, but not sure why
    min_bin_support = 3

    smoothed = {
        label: savgol_filter(np.array(curve), window_length, polyorder)
        for label, curve in data_dict.items()
    }

    all_vals = np.concatenate(list(smoothed.values()))
    global_max = np.max(all_vals)
    y_cutoff = cutoff_frac * global_max

    # Track per-pair crossings
    pairwise_crossings = []  # Each entry: (pair, idx, deviation)

    for (label1, y1), (label2, y2) in itertools.combinations(smoothed.items(), 2):
        diff = y1 - y2
        sign_change = np.where(np.diff(np.sign(diff)))[0]

        for idx in sign_change:
            if y1[idx] > y_cutoff and y2[idx] > y_cutoff:
                deviation = abs(idx)  # Deviation is the x-coordinate (index) difference
                pairwise_crossings.append(((label1, label2), idx, deviation))

    if not pairwise_crossings:
        raise ValueError("No valid crossings found.")

    # Cluster crossings by bin to find the consensus region
    bin_counter = Counter()
    crossing_by_bin = {}

    for pair, idx, deviation in pairwise_crossings:
        bin_id = idx // bin_width
        bin_counter[bin_id] += 1
        crossing_by_bin.setdefault(bin_id, []).append((pair, idx, deviation))

    best_bin, count = bin_counter.most_common(1)[0]
    if count < min_bin_support:
        raise ValueError("Insufficient support in any crossing cluster.")

    # Keep only one crossing per pair — the one closest to the bin center
    crossings_in_bin = crossing_by_bin[best_bin]
    bin_center = best_bin * bin_width + bin_width // 2

    closest_crossing_per_pair = {}
    for pair, idx, deviation in crossings_in_bin:
        if pair not in closest_crossing_per_pair:
            closest_crossing_per_pair[pair] = (idx, deviation)
        else:
            prev_idx, prev_deviation = closest_crossing_per_pair[pair]
            if abs(idx - bin_center) < abs(prev_idx - bin_center):
                closest_crossing_per_pair[pair] = (idx, deviation)

    # Collect the final indices and deviations
    chosen_indices = [idx for idx, _ in closest_crossing_per_pair.values()]
    chosen_deviations = [abs(idx - bin_center) for idx, _ in closest_crossing_per_pair.values()]

    avg_index = int(round(np.mean(chosen_indices)))
    avg_deviation = float(np.mean(chosen_deviations))

    return avg_index, avg_deviation

def plot_derivative_of_mean(ax, filenames, type, ylabel, logscale=False, cumsum=False):
    cmap = get_cmap("tab10")
    ax2 = ax.twinx()

    for i, filename in enumerate(filenames):
        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)

        if type == "epilength":
            data = epilength_all_trials[filename]
        elif type == "length":
            data = length_all_trials[filename]
        elif type == "testing_epilength":
            data = testing_epilength_all_trials[filename]

        mean_array = np.mean(data, axis=0)
        derivative = np.gradient(mean_array)

        # Determine valid window length for Savitzky-Golay filter
        polyorder = 2
        max_window = len(derivative) if len(derivative) % 2 == 1 else len(derivative) - 1
        window_length = min(21, max_window)
        if window_length <= polyorder:
            smoothed_derivative = derivative  # fallback if smoothing not possible
        else:
            smoothed_derivative = savgol_filter(derivative, window_length, polyorder, mode='interp')

        if cumsum:
            smoothed_derivative = np.cumsum(smoothed_derivative)

        episode_num = np.arange(1, len(derivative) + 1)
        color = cmap(i % cmap.N)

        ax.plot(episode_num, mean_array, label=f"Mean: {reset_rate}, {epsilon}, {learning_rate}, {gamma}",
                linestyle='--', color=color)
        ax2.plot(episode_num, smoothed_derivative,
                 label=f"1st Deriv: {reset_rate}, {epsilon}, {learning_rate}, {gamma}",
                 linestyle='-', color=color)

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax2.set_ylabel("1st Derivative")
    ax2.tick_params(axis='y')

# Function to plot average data with error bars and intersection point for "epilength"
def plot_avg_with_error_bars(ax, filenames, type, ylabel, logscale=False, cumsum=False):
    data_dict = {}  # Dictionary to store data for finding intersection points
    color_mapping = {}  # Map reset rates to colors for consistent text coloring

    for i, filename in enumerate(filenames):
        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]  # Remove exactly 4 characters from the end

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)

        if type == "epilength":
            data = epilength_all_trials[filename]
            data_dict[filename] = np.mean(data, axis=0)  # Store mean data for intersection calculation
        elif type == "length":
            data = length_all_trials[filename]
        elif type == "testing_epilength":
            data = testing_epilength_all_trials[filename]

        # Calculate the average and standard deviation
        avg_array = np.mean(data, axis=0)
        std_array = np.std(data, axis=0)  # Compute standard deviation across trials

        episode_num = np.linspace(1, avg_array.shape[0], avg_array.shape[0])

        if cumsum:
            # Compute cumulative sum and cumulative error
            avg_array = np.cumsum(avg_array)
            std_array = np.sqrt(np.cumsum(std_array ** 2))

        # Determine indices to plot 10 error bars evenly spaced
        error_bar_indices = np.linspace(0, len(episode_num) - 1, 10, dtype=int)

        # Select a distinct color from the color cycle
        color = color_cycle[i % 20]
        color_mapping[reset_rate] = color  # Store color for reset rate

        # Plot the main data line
        line, = ax.plot(episode_num, avg_array, label=f"{reset_rate}, {epsilon}, {learning_rate}, {gamma}", color=color)

    # If the data type is "epilength", find and plot the intersection point
    if len(filenames) > 1:
        # Plot error bars with the same color as the line -- this is a bit hacky
        ax.errorbar(episode_num[error_bar_indices], avg_array[error_bar_indices], yerr=std_array[error_bar_indices], 
                    fmt='none', ecolor=color, capsize=3)
        if type == "epilength" and data_dict:
            if not cumsum: 
                try:
                    intersection_index, intersection_dist = find_intersection_point(data_dict)
                    if intersection_index != -1:
                        # Get the x and y coordinates of the intersection point
                        intersection_x = intersection_index + 1  # Adjust for 1-based episode numbering
                        intersection_y = np.mean([data[intersection_index] for data in data_dict.values()])

                        # Plot the intersection point as a dot
                        ax.plot(intersection_x, intersection_y, 'ro', label="Intersection Point")

                        # Add a text box with the intersection coordinates
                        ax.text(intersection_x, intersection_y, f"({intersection_x}, {intersection_y:.2f})", 
                                fontsize=8, color='red', ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

                        # Calculate the fraction of cumulative steps taken before the intersection point for each resetting rate
                        fractions = {}
                        for filename, data in data_dict.items():
                            reset_rate, _, _, _ = extract_resetting_rate_epsilon_learning_rate_gamma(filename)
                            cumulative_steps = np.cumsum(data)
                            total_steps = cumulative_steps[-1]
                            fraction = cumulative_steps[intersection_index] / total_steps
                            fractions[reset_rate] = fraction

                        # Add fraction text only on loglog plot, using matching colors
                        if logscale:
                            text_y = 0.05  # Start at the bottom
                            for reset_rate, fraction in sorted(fractions.items()):
                                color = color_mapping[reset_rate]  # Get matching color
                                ax.text(0.02, text_y, f"r={reset_rate}: {fraction:.2%}", 
                                        transform=ax.transAxes, fontsize=8, color=color, ha='left', va='bottom', 
                                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                                text_y += 0.05  # Move up for the next text box
                except Exception as e:
                    print(f"Skipping intersection point: {e}")

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

# Function to plot five sample paths from the data
def plot_sample_paths(ax, filenames, type, ylabel, logscale=False, cumsum=False, num_samples=5):
    for i, filename in enumerate(filenames):
        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)

        if type == "epilength":
            data = epilength_all_trials[filename]
        elif type == "length":
            data = length_all_trials[filename]
        elif type == "testing_epilength":
            data = testing_epilength_all_trials[filename]
        else:
            raise ValueError(f"Unknown type: {type}")

        num_trials = data.shape[0]
        num_episodes = data.shape[1]

        # Select sample indices evenly spaced among available trials
        if num_trials < num_samples:
            sample_indices = np.arange(num_trials)
        else:
            sample_indices = np.linspace(0, num_trials - 1, num_samples, dtype=int)

        # Plot each sample path
        for idx, trial_idx in enumerate(sample_indices):
            sample_path = data[trial_idx]
            if cumsum:
                sample_path = np.cumsum(sample_path)

            episode_num = np.linspace(1, num_episodes, num_episodes)
            color = color_cycle[(i * num_samples + idx) % len(color_cycle)]
            ax.plot(episode_num, sample_path, label=f"Trial {trial_idx} (r={reset_rate})", color=color, alpha=0.7)

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

def plot_time_series_distribution(ax, filenames, type, ylabel, logscale=False):
    for i, filename in enumerate(filenames):
        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)

        if type == "epilength":
            data = epilength_all_trials[filename]
        elif type == "length":
            data = length_all_trials[filename]
        elif type == "testing_epilength":
            data = testing_epilength_all_trials[filename]

        episode_num = np.linspace(1, data.shape[1], data.shape[1])
        distribution_color = color_cycle[i % 20]
        tau_array = []

        for j, episode in enumerate(episode_num):
            episode_data = data[:, j]
            hist, bin_edges = np.histogram(episode_data, bins=50, density=True)
            probabilities = hist / np.max(hist)

            for k in range(len(bin_edges) - 1):
                ax.fill_betweenx(
                    [bin_edges[k], bin_edges[k + 1]],
                    episode - 0.5,
                    episode + 0.5,
                    color=distribution_color,
                    alpha=probabilities[k] * 0.5
                )

            # Fit exponential decay to the histogram
            # bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            # try:
            #     popt, _ = curve_fit(exponential, bin_centers, hist, p0=(1.0, np.mean(episode_data)))
            #     A_fit, tau_fit = popt
            #     tau_array.append(tau_fit)
            # except RuntimeError:
            #     tau_array.append(np.nan)  # Append NaN if fit fails

        # Plot mean and median
        mean_array = np.mean(data, axis=0)
        median_array = np.median(data, axis=0)
        ax.plot(episode_num, mean_array, label=f"Mean: {reset_rate}, {epsilon}, {learning_rate}, {gamma}", color="blue", linestyle='--')
        ax.plot(episode_num, median_array, label=f"Median: {reset_rate}, {epsilon}, {learning_rate}, {gamma}", color="orange", linestyle='--')

        # # Plot decay constant (tau) on twin axis
        # ax2 = ax.twinx()
        # ax2.plot(episode_num, tau_array, color='green', label='Decay time (τ)')
        # ax2.set_ylabel("Decay time τ", fontsize='small', color='green')
        # ax2.tick_params(axis='y', labelcolor='green', labelsize=8)

        # --- Plot vertical line at max std/mean index ---
        fig_temp, ax_temp = plt.subplots()
        max_indices = plot_std_over_mean(ax_temp, [filename], type, ylabel, logscale=logscale, cumsum=False, plot_maxima=True)
        plt.close(fig_temp)

        if filename in max_indices:
            max_idx = max_indices[filename]
            ax.axvline(episode_num[max_idx], color='r', linestyle='--', label=f"Max std/mean index: {reset_rate}, {epsilon}, {learning_rate}, {gamma}")

        ax.set_title(ylabel, fontsize=8)
        ax.set_xlabel("Episode number")
        ax.set_ylabel("Episode length", fontsize='small')

        if logscale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ax.set_xlim(0, 300)

def plot_median(ax, filenames, type, ylabel, logscale=False, cumsum=False):
    data_dict = {}  # Dictionary to store data for finding intersection points
    color_mapping = {}  # Map reset rates to colors for consistent text coloring

    for i, filename in enumerate(filenames):
        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]  # Remove exactly 4 characters from the end

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)

        if type == "epilength":
            data = epilength_all_trials[filename]
            data_dict[filename] = np.median(data, axis=0)
        elif type == "length":
            data = length_all_trials[filename]
        elif type == "testing_epilength":
            data = testing_epilength_all_trials[filename]

        median_array = np.median(data, axis=0)  # Compute the median across trials
        if cumsum:
            median_array = np.cumsum(median_array)  # Compute cumulative sum

        # Select a distinct color from the color cycle
        color = color_cycle[i % 20]
        color_mapping[reset_rate] = color  # Store color for reset rate

        episode_num = np.linspace(1, median_array.shape[0], median_array.shape[0])

        # Plot the main data line
        ax.plot(episode_num, median_array, label=f"{reset_rate}, {epsilon}, {learning_rate}, {gamma}", color=color)

    # If the data type is "epilength", find and plot the intersection point
    if type == "epilength" and data_dict:
        if not cumsum: 
            intersection_index, intersection_dist = find_intersection_point(data_dict)
            if intersection_index != -1:
                # Get the x and y coordinates of the intersection point
                intersection_x = intersection_index + 1  # Adjust for 1-based episode numbering
                intersection_y = np.mean([data[intersection_index] for data in data_dict.values()])

                # Plot the intersection point as a dot
                ax.plot(intersection_x, intersection_y, 'ro', label="Intersection Point")

                # Add a text box with the intersection coordinates
                ax.text(intersection_x, intersection_y, f"({intersection_x}, {intersection_y:.2f})", 
                        fontsize=8, color='red', ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

                # Calculate the fraction of cumulative steps taken before the intersection point for each resetting rate
                fractions = {}
                for filename, data in data_dict.items():
                    reset_rate, _, _, _ = extract_resetting_rate_epsilon_learning_rate_gamma(filename)
                    cumulative_steps = np.cumsum(data)
                    total_steps = cumulative_steps[-1]
                    fraction = cumulative_steps[intersection_index] / total_steps
                    fractions[reset_rate] = fraction

                # Only add the fraction text box on the loglog plot, with colors matching their respective curves
                if logscale:
                    text_y = 0.05  # Start at the bottom
                    for reset_rate, fraction in sorted(fractions.items()):
                        color = color_mapping[reset_rate]  # Get matching color
                        ax.text(0.02, text_y, f"r={reset_rate}: {fraction:.2%}", 
                                transform=ax.transAxes, fontsize=8, color=color, ha='left', va='bottom', 
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                        text_y += 0.05  # Move up for the next text box

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')


def plot_max(ax, filenames, type, ylabel, logscale=False, cumsum=False, smoothing_window=10):
    data_dict = {}  # Dictionary to store data for finding intersection points

    for i, filename in enumerate(filenames):
        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]  

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)

        if type == "epilength":
            data = epilength_all_trials[filename]
            data_dict[filename] = np.max(data, axis=0)
        elif type == "length":
            data = length_all_trials[filename]
        elif type == "testing_epilength":
            data = testing_epilength_all_trials[filename]

        max_array = np.max(data, axis=0)
        if cumsum:
            max_array = np.cumsum(max_array)

        smoothed_max_array = np.convolve(max_array, np.ones(smoothing_window) / smoothing_window, mode='valid')
        episode_num = np.linspace(1, len(smoothed_max_array), len(smoothed_max_array))

        color = color_cycle[i % 20]  # Assign a distinct color
        ax.plot(episode_num, smoothed_max_array, label=f"{reset_rate}, {epsilon}, {learning_rate}, {gamma}", color=color)

    # If the data type is "epilength", find and plot the intersection point
    if type == "epilength" and data_dict:
        if not cumsum: 
            try: 
                intersection_index, intersection_dist = find_intersection_point(data_dict)
                # print("intersection_index", intersection_index)
            except ValueError as e:
                print(f"Error: {e}")
            if intersection_index != -1:
                intersection_x = intersection_index + 1  # Adjust for 1-based episode numbering
                intersection_y = np.mean([data[intersection_index] for data in data_dict.values()])

                # Plot the intersection point as a dot
                ax.plot(intersection_x, intersection_y, 'ro', label="Intersection Point")

                # Add a text box with the intersection coordinates
                ax.text(intersection_x, intersection_y, f"({intersection_x}, {intersection_y:.2f})", 
                        fontsize=8, color='red', ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

                # Calculate the fraction of cumulative steps taken before the intersection point for each resetting rate
                fractions = {}
                for filename, data in data_dict.items():
                    reset_rate, _, _, _ = extract_resetting_rate_epsilon_learning_rate_gamma(filename)
                    cumulative_steps = np.cumsum(data)
                    total_steps = cumulative_steps[-1]
                    fraction = cumulative_steps[intersection_index] / total_steps
                    fractions[reset_rate] = fraction

                # Only add the fraction text box on the loglog plot
                if logscale:
                    text_y = 0.05  # Start at the bottom
                    for filename, (reset_rate, fraction) in zip(data_dict.keys(), sorted(fractions.items())):
                        line_color = ax.get_lines()[list(data_dict.keys()).index(filename)].get_color()
                        ax.text(0.02, text_y, f"r={reset_rate}: {fraction:.2%}", 
                                transform=ax.transAxes, fontsize=8, color=line_color, ha='left', va='bottom', 
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                        text_y += 0.05  # Move up for the next text box

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

def plot_mean_minus_median(ax, filenames, type, ylabel, logscale=False, cumsum=False):
    for i, filename in enumerate(filenames):
        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)

        if type == "epilength":
            data = epilength_all_trials[filename]
        elif type == "length":
            data = length_all_trials[filename]
        elif type == "testing_epilength":
            data = testing_epilength_all_trials[filename]

        mean_array = np.mean(data, axis=0)
        median_array = np.median(data, axis=0)

        mean_minus_median = mean_array - median_array

        if cumsum:
            mean_minus_median = np.cumsum(mean_minus_median)

        episode_num = np.linspace(1, len(mean_minus_median), len(mean_minus_median))

        color = color_cycle[i % 20]  # Assign a distinct color
        ax.plot(episode_num, mean_minus_median, label=f"{reset_rate}, {epsilon}, {learning_rate}, {gamma}", color=color)

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')


def plot_mean_over_median(ax, filenames, type, ylabel, logscale=False, cumsum=False):
    for i, filename in enumerate(filenames):
        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)

        if type == "epilength":
            data = epilength_all_trials[filename]
        elif type == "length":
            data = length_all_trials[filename]
        elif type == "testing_epilength":
            data = testing_epilength_all_trials[filename]

        mean_array = np.mean(data, axis=0)
        median_array = np.median(data, axis=0)

        mean_over_median = np.divide(mean_array, median_array, out=np.zeros_like(mean_array), where=median_array != 0)

        if cumsum:
            mean_over_median = np.cumsum(mean_over_median)

        episode_num = np.linspace(1, len(mean_over_median), len(mean_over_median))

        color = color_cycle[i % 20]  # Assign a distinct color
        ax.plot(episode_num, mean_over_median, label=f"{reset_rate}, {epsilon}, {learning_rate}, {gamma}", color=color)

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

def plot_min(ax, filenames, type, ylabel, logscale=False, cumsum=False, smoothing_window=10):
    for i, filename in enumerate(filenames):
        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]  

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)

        if type == "epilength":
            data = epilength_all_trials[filename]
        elif type == "length":
            data = length_all_trials[filename]
        elif type == "testing_epilength":
            data = testing_epilength_all_trials[filename]

        min_array = np.min(data, axis=0)
        if cumsum:
            min_array = np.cumsum(min_array)

        smoothed_min_array = np.convolve(min_array, np.ones(smoothing_window) / smoothing_window, mode='valid')
        episode_num = np.linspace(1, len(smoothed_min_array), len(smoothed_min_array))

        color = color_cycle[i % 20]  # Assign a distinct color
        ax.plot(episode_num, smoothed_min_array, label=f"{reset_rate}, {epsilon}, {learning_rate}, {gamma}", color=color)

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

def plot_std(ax, filenames, type, ylabel, logscale=False, cumsum=False, smoothing_window=10):
    data_dict = {}  # Dictionary to store data for finding intersection points
    for i, filename in enumerate(filenames):
        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]
        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)
        if type == "epilength":
            data = epilength_all_trials[filename]
            data_dict[filename] = np.std(data, axis=0)  # Store std data for intersection calculation
        elif type == "length":
            data = length_all_trials[filename]
        elif type == "testing_epilength":
            data = testing_epilength_all_trials[filename]
        std_array = np.std(data, axis=0)  # Compute standard deviation across trials
        if cumsum:
            std_array = np.cumsum(std_array)
        # Smooth the standard deviation using a moving average
        smoothed_std_array = np.convolve(std_array, np.ones(smoothing_window) / smoothing_window, mode='valid')
        episode_num = np.linspace(1, len(smoothed_std_array), len(smoothed_std_array))
        color = color_cycle[i % 20]  # Assign a distinct color
        ax.plot(episode_num, smoothed_std_array, label=f"{reset_rate}, {epsilon}, {learning_rate}, {gamma}", color=color)

    # If the data type is "epilength", find and plot the intersection point
    if type == "epilength" and data_dict:
        if not cumsum: 
            try: 
                intersection_index, intersection_dist = find_intersection_point(data_dict)
                # print("intersection_index", intersection_index)
            except ValueError as e:
                print(f"Error: {e}")
            if intersection_index != -1:
                intersection_x = intersection_index + 1  # Adjust for 1-based episode numbering
                intersection_y = np.mean([data[intersection_index] for data in data_dict.values()])

                # Plot the intersection point as a dot
                ax.plot(intersection_x, intersection_y, 'ro', label="Intersection Point")

                # Add a text box with the intersection coordinates
                ax.text(intersection_x, intersection_y, f"({intersection_x}, {intersection_y:.2f})", 
                        fontsize=8, color='red', ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

                # Calculate the fraction of cumulative steps taken before the intersection point for each resetting rate
                fractions = {}
                for filename, data in data_dict.items():
                    reset_rate, _, _, _ = extract_resetting_rate_epsilon_learning_rate_gamma(filename)
                    cumulative_steps = np.cumsum(data)
                    total_steps = cumulative_steps[-1]
                    fraction = cumulative_steps[intersection_index] / total_steps
                    fractions[reset_rate] = fraction

                # Only add the fraction text box on the loglog plot
                if logscale:
                    text_y = 0.05  # Start at the bottom
                    for filename, (reset_rate, fraction) in zip(data_dict.keys(), sorted(fractions.items())):
                        line_color = ax.get_lines()[list(data_dict.keys()).index(filename)].get_color()
                        ax.text(0.02, text_y, f"r={reset_rate}: {fraction:.2%}", 
                                transform=ax.transAxes, fontsize=8, color=line_color, ha='left', va='bottom', 
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                        text_y += 0.05  # Move up for the next text box

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

def plot_std_over_mean(ax, filenames, type, ylabel, logscale=False, cumsum=False, smoothing_window=10, plot_maxima=False):
    maxima_indices = {}  # Dictionary to store maxima indices for each filename
    avg_values = []  # Store average std/mean values for annotation

    for i, filename in enumerate(filenames):
        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]  

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)

        if type == "epilength":
            data = epilength_all_trials[filename]
        elif type == "length":
            data = length_all_trials[filename]
        elif type == "testing_epilength":
            data = testing_epilength_all_trials[filename]

        mean_array = np.mean(data, axis=0)
        std_array = np.std(data, axis=0)

        std_over_mean = np.divide(std_array, mean_array, out=np.zeros_like(std_array), where=mean_array != 0)
        # std_over_mean = std_array

        if cumsum:
            std_over_mean = np.cumsum(std_over_mean)

        smoothed_std_over_mean = np.convolve(std_over_mean, np.ones(smoothing_window) / smoothing_window, mode='valid')
        episode_num = np.linspace(1, len(smoothed_std_over_mean), len(smoothed_std_over_mean))

        color = color_cycle[i % 20]  # Assign a distinct color

        if plot_maxima and type == "epilength" and cumsum == False:
            max_index = np.argmax(smoothed_std_over_mean)
            maxima_indices[filename] = max_index
            ax.axvline(x=max_index + 1, color='red', linestyle='--', alpha=0.7, label=f"Max: {reset_rate}, {epsilon}")

        line, = ax.plot(episode_num, smoothed_std_over_mean, label=f"{reset_rate}, {epsilon}, {learning_rate}, {gamma}", color=color)

        avg_std_over_mean = np.mean(std_over_mean)
        avg_values.append((avg_std_over_mean, color))

    # text to display the average std/mean, which is no longer relevant
    # if logscale:
    #     text_x = 0.05
    #     text_y_start = 0.95
    #     line_spacing = 0.05

    #     for i, (avg, color) in enumerate(avg_values):
    #         ax.text(text_x, text_y_start - i * line_spacing, f"avg: {avg:.3f}", 
    #                 color=color, fontsize=8, ha='left', va='top', transform=ax.transAxes)

    ax.set_title(ylabel, fontsize=8)
    ax.set_xlabel("Episode number")
    ax.set_ylabel(ylabel, fontsize='small')
    
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    return maxima_indices


def plot_distribution(filenames, type, episodes_array, ylabel):
    reset_rates = sorted(set(
        extract_resetting_rate_epsilon_learning_rate_gamma(f[:-4])[0] for f in filenames if f.endswith('.npz')
    ))  # Extract and sort unique reset rates

    num_cols = len(episodes_array)
    num_rows = 1
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5), sharey=False)
    
    if num_cols == 1:
        axs = [axs]
    
    all_handles, all_labels = [], []
    
    for filename in filenames:
        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]

        reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)
        
        if type == "epilength":
            data = epilength_all_trials[filename]
        elif type == "length":
            data = length_all_trials[filename]
        elif type == "testing_epilength":
            data = testing_epilength_all_trials[filename]

        for i, episode in enumerate(episodes_array):
            if episode >= data.shape[1]:
                continue

            data_per_episode = data[:, episode]
            ax = axs[i]
            counts, bins, patches = ax.hist(data_per_episode, bins=20, alpha=0.7, label=f"Reset Rate: {reset_rate}, Epsilon: {epsilon}, Gamma: {gamma}", density=True)

            bin_centers = 0.5 * (bins[1:] + bins[:-1])

            # Fit exponential
            try:
                popt, pcov = curve_fit(exponential, bin_centers, counts, p0=(1.0, np.mean(data_per_episode)))
                A_fit, tau_fit = popt
                x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 300)
                y_fit = exponential(x_fit, *popt)
                ax.plot(x_fit, y_fit, linestyle='--', linewidth=1.5, color='black')

                # Annotate with fitting parameters
                textstr = f"$\\tau = {tau_fit:.2f}$"
                ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=8,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

                print(f"[{filename_stripped}, Episode {episode}] Exponential Fit: tau = {tau_fit:.3f}")
            except RuntimeError:
                print(f"[{filename_stripped}, Episode {episode}] Fit failed.")

            ax.set_title(f"Episode: {episode}", fontsize=10)
            ax.set_xlabel(ylabel, fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)

            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)
    
    unique_labels = dict(zip(all_labels, all_handles))
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    base_filename = get_base_filename(filenames[0])
    plt.savefig(f"figs/fpt_dist_{type}_episodes_{'_'.join(map(str, episodes_array))}_{base_filename}.png", bbox_inches='tight')

# def plot_distribution(filenames, type, episodes_array, ylabel):
#     reset_rates = sorted(set(
#         extract_resetting_rate_epsilon_learning_rate_gamma(f[:-4])[0] for f in filenames if f.endswith('.npz')
#     ))  # Extract and sort unique reset rates

#     num_cols = len(episodes_array)
#     num_rows = 1  # Set to 1 row for all columns
    
#     fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5), sharey=True)
    
#     if num_cols == 1:
#         axs = [axs]  # Ensure axs is iterable for single-column cases
    
#     all_handles, all_labels = [], []
    
#     for filename in filenames:
#         if filename.endswith('.npz'):
#             filename_stripped = filename[:-4]
        
#         reset_rate, epsilon, learning_rate, gamma = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)
        
#         if type == "epilength":
#             data = epilength_all_trials[filename]
#         elif type == "length":
#             data = length_all_trials[filename]
#         elif type == "testing_epilength":
#             data = testing_epilength_all_trials[filename]

#         for i, episode in enumerate(episodes_array):
#             if episode >= data.shape[1]:
#                 continue  # Skip if the requested episode index is out of range

#             data_per_episode = data[:, episode]  # Extract values at the given episode number
            
#             ax = axs[i]
#             ax.hist(data_per_episode, bins=20, alpha=0.7, label=f"Reset Rate: {reset_rate}, Epsilon: {epsilon}, Gamma: {gamma}")
#             ax.set_title(f"Episode: {episode}", fontsize=10)
#             ax.set_xlabel(ylabel, fontsize=8)
#             ax.set_ylabel("Frequency", fontsize=8)
#             ax.tick_params(axis='both', which='major', labelsize=8)
            
#             handles, labels = ax.get_legend_handles_labels()
#             all_handles.extend(handles)
#             all_labels.extend(labels)
    
#     # Remove duplicate labels
#     unique_labels = dict(zip(all_labels, all_handles))
#     fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize='small')
    
#     plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to reduce top white space
#     base_filename = get_base_filename(filenames[0])
#     plt.savefig(f"figs/fpt_dist_{type}_episodes_{'_'.join(map(str, episodes_array))}_{base_filename}.png", bbox_inches='tight')


# Function to create 1x3 subplots for a single metric
# this creates 4 plots, each with 3 subplots (mean, median, max, min) for the specified metric
def create_1x3_subplots(filenames, datatype, ylabel, filename_prefix, type):

    # different plotting options depending on the type of data we look at
    if type == 'mean':
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        plot_avg_with_error_bars(axs[0], filenames, datatype, f"{ylabel}", logscale=True, cumsum=False)
        plot_avg_with_error_bars(axs[1], filenames, datatype, f"{ylabel}", logscale=False, cumsum=False)
        plot_avg_with_error_bars(axs[2], filenames, datatype, f"{ylabel} (cumulative sum)", logscale=False, cumsum=True)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    elif type == 'std_over_mean':
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        max_index = plot_std_over_mean(axs[0], filenames, datatype, f"{ylabel}", logscale=True, cumsum=False, plot_maxima=True)
        # print('max index', max_index)
        plot_std_over_mean(axs[1], filenames, datatype, f"{ylabel}", logscale=False, cumsum=False, plot_maxima=True)
        # plot_std_over_mean(axs[2], filenames, datatype, f"{ylabel} (cumulative sum)", logscale=False, cumsum=True, plot_maxima=True)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    elif type == 'std':
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        plot_std(axs[0], filenames, datatype, f"{ylabel}", logscale=True, cumsum=False)
        plot_std(axs[1], filenames, datatype, f"{ylabel}", logscale=False, cumsum=False)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    elif type == 'median':
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        plot_median(axs[0], filenames, datatype, f"{ylabel}", logscale=True, cumsum=False)
        plot_median(axs[1], filenames, datatype, f"{ylabel}", logscale=False, cumsum=False)
        # plot_median(axs[2], filenames, datatype, f"{ylabel} (cumulative sum)", logscale=False, cumsum=True)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    elif type == "mean_minus_median":
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        plot_mean_minus_median(axs[0], filenames, datatype, f"{ylabel}", logscale=True, cumsum=False)
        plot_mean_minus_median(axs[1], filenames, datatype, f"{ylabel}", logscale=False, cumsum=False)
        # plot_mean_minus_median(axs[2], filenames, datatype, f"{ylabel} (cumulative sum)", logscale=False, cumsum=True)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    elif type == "mean_over_median":
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        plot_mean_over_median(axs[0], filenames, datatype, f"{ylabel}", logscale=True, cumsum=False)
        plot_mean_over_median(axs[1], filenames, datatype, f"{ylabel}", logscale=False, cumsum=False)
        plot_mean_over_median(axs[2], filenames, datatype, f"{ylabel} (cumulative sum)", logscale=False, cumsum=True)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    elif type == 'max':
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        plot_max(axs[0], filenames, datatype, f"{ylabel}", logscale=True, cumsum=False)
        plot_max(axs[1], filenames, datatype, f"{ylabel}", logscale=False, cumsum=False)
        # plot_max(axs[2], filenames, datatype, f"{ylabel} (cumulative sum)", logscale=False, cumsum=True)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    elif type == 'min':
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        plot_min(axs[0], filenames, datatype, f"{ylabel}", logscale=True, cumsum=False)
        plot_min(axs[1], filenames, datatype, f"{ylabel}", logscale=False, cumsum=False)
        # plot_min(axs[2], filenames, datatype, f"{ylabel} (cumulative sum)", logscale=False, cumsum=True)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    elif type == 'time_series_full':
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        plot_time_series_distribution(axs[0], filenames, datatype, f"{ylabel}", logscale=True)
        plot_time_series_distribution(axs[1], filenames, datatype, f"{ylabel}", logscale=False)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    elif type == "derivative":
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_derivative_of_mean(ax, filenames, datatype, f"{ylabel}", logscale=True, cumsum=False)

        # Add a single legend outside the plot
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')
    
    elif type == "sample_paths":
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        plot_sample_paths(axs[0], filenames, datatype, f"{ylabel}", logscale=True, cumsum=False)
        plot_sample_paths(axs[1], filenames, datatype, f"{ylabel}", logscale=False, cumsum=False)

        # Add a single legend outside the plots
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    # Add a single legend outside the plots
    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    # Save the plot
    base_filename = get_base_filename(filenames[0]) # this does just make it save to the first filename, but it's ok since everything is cropped away anyways
    plt.savefig(f"figs/{filename_prefix}_{base_filename}.png", bbox_inches='tight')

# Check if filenames only have one unique epsilon -- otherwise the line plots do not really make sense
unique_epsilons = set(
    extract_resetting_rate_epsilon_learning_rate_gamma(f[:-4])[1] for f in filenames if f.endswith('.npz')
)

# create metric vs episode num, multiple lines per plot plots
if len(unique_epsilons) == 1:
    # call plotting function for each metric
    create_1x3_subplots(filenames, "length", "Avg Final Path Length", "avg_final_path_length", type='mean')
    create_1x3_subplots(filenames, "testing_epilength", "Avg Testing Epilength", "avg_testing_epilength", type='mean')
    create_1x3_subplots(filenames, "epilength", "Avg Length", "avg_length", type='mean')

    #======= option to plot training and testing epilength together -- uncomment if needed (also a bit hacky) =======
    #Plot both "testing_epilength" and "epilength" on the same plot, loglog and not cumsum

    # fig, ax = plt.subplots(figsize=(10, 6))
    # # Plot "testing_epilength" with a distinct color and label
    # plot_avg_with_error_bars(ax, filenames, "testing_epilength", "Avg Testing Epilength", logscale=True, cumsum=False)
    # for line in ax.get_lines()[-1:]:  # Get the last line added (testing_epilength)
    #     line.set_color('blue')
    #     line.set_label("Avg Testing Epilength")
    # # Plot "epilength" with a distinct color and label
    # plot_avg_with_error_bars(ax, filenames, "epilength", "Avg Training Length", logscale=True, cumsum=False)
    # for line in ax.get_lines()[-1:]:  # Get the last line added (epilength)
    #     line.set_color('green')
    #     line.set_label("Avg Training Length")
    # plot_avg_with_error_bars(ax, filenames, "length", "Avg Final Length", logscale=True, cumsum=False)
    # for line in ax.get_lines()[-1:]:  # Get the last line added
    #     line.set_color('purple')
    #     line.set_label("Avg Final Length")
    
    # ax.legend(loc='best', fontsize='small')
    # ax.set_title("Avg Testing Epilength, Avg Training Length (Log-Log, Not Cumulative)", fontsize=10)
    # base_filename = get_base_filename(filenames[0])
    # plt.savefig(f"figs/combined_avg_testing_epilength_avg_length_avg_final_path_length_std_over_mean_loglog_{base_filename}.png", bbox_inches='tight')
    # plt.show()

    #======= end of option to plot training and testing epilength together =======

    create_1x3_subplots(filenames, "length", "Median Final Path Length", "median_final_path_length", type='median')
    create_1x3_subplots(filenames, "testing_epilength", "Median Testing Epilength", "median_testing_epilength", type='median')
    create_1x3_subplots(filenames, "epilength", "Median Length", "median_length", type='median')

    create_1x3_subplots(filenames, "length", "Maximum Final Path Length", "max_final_path_length", type='max')
    create_1x3_subplots(filenames, "testing_epilength", "Maximum Testing Epilength", "max_testing_epilength", type='max')
    create_1x3_subplots(filenames, "epilength", "Maximum Length", "max_length", type='max')

    # create_1x3_subplots(filenames, "length", "Minimum Final Path Length", "min_final_path_length", type='min')
    # create_1x3_subplots(filenames, "testing_epilength", "Minimum Testing Epilength", "min_testing_epilength", type='min')
    # create_1x3_subplots(filenames, "epilength", "Minimum Length", "min_length", type='min')

    # create_1x3_subplots(filenames, "length", "Std/Mean Final Path Length", "std_over_mean_final_path_length", type='std_over_mean')
    # create_1x3_subplots(filenames, "testing_epilength", "Std/Mean Testing Epilength", "std_over_mean_testing_epilength", type='std_over_mean')
    # create_1x3_subplots(filenames, "epilength", "Std/Mean Length", "std_over_mean_length", type='std_over_mean')

    create_1x3_subplots(filenames, "length", "Std Final Path Length", "std_final_path_length", type='std')
    create_1x3_subplots(filenames, "testing_epilength", "Std Testing Epilength", "std_testing_epilength", type='std')
    create_1x3_subplots(filenames, "epilength", "Std Length", "std_length", type='std')

    # five sample paths
    # create_1x3_subplots(filenames, "epilength", "Sample Training Episode Paths", "sample_paths_episode_length", type='sample_paths')

    # time series
    # create_1x3_subplots(filenames, "epilength", "Binned Distribution, Mean and Median", "time_series_full_prob_dist", type='time_series_full')

    # mean minus median and mean over median
    # create_1x3_subplots(filenames, "length", "Mean - Median Final Path Length", "mean_minus_median_final_path_length", type='mean_minus_median')
    # create_1x3_subplots(filenames, "testing_epilength", "Mean - Median Testing Epilength", "mean_minus_median_testing_epilength", type='mean_minus_median')
    # create_1x3_subplots(filenames, "epilength", "Mean - Median Length", "mean_minus_median_length", type='mean_minus_median')

    # create_1x3_subplots(filenames, "length", "Mean / Median Final Path Length", "mean_over_median_final_path_length", type='mean_over_median')
    # create_1x3_subplots(filenames, "testing_epilength", "Mean / Median Testfing Epilength", "mean_over_median_testing_epilength", type='mean_over_median')
    # create_1x3_subplots(filenames, "epilength", "Mean / Median Length", "mean_over_median_length", type='mean_over_median')

    # derivatives
    # create_1x3_subplots(filenames, "epilength", "Derivative of Mean Final Path Length", "derivative_mean_final_path_length", type='derivative')

    # plot distribution of episode lengths at a specific episode number, eg. 0 25 50
    # plot_distribution(filenames, "epilength", [0, 100, 200], "Episode Length Distribution")

else:
    print("Skipping plots as filenames contain multiple epsilon values.")


# --------------------------------------
# 2D heatmap functions & plotting
# --------------------------------------

# define a function to plot the heatmap with contour lines
# options: epilength, length, testing_epilength, fpt
def plot_heatmap_avg(filenames, ylabel, type):
    heatmap_data = {}
    epsilons = set()
    reset_rates = set()

    for filename in filenames:
        try:
            if filename.endswith('.npz'):
                filename_stripped = filename[:-4]
                reset_rate, epsilon, _, _ = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)
                
                if filename in epilength_all_trials and type == "epilength":
                    data = epilength_all_trials[filename]
                    heatmap_data[(reset_rate, epsilon)] = np.mean(np.cumsum(data, axis=1)[:, -1])  # Avg cumulative length
                elif filename in length_all_trials and type == "length":
                    data = length_all_trials[filename]
                    heatmap_data[(reset_rate, epsilon)] = np.mean(np.cumsum(data, axis=1)[:, -1])  # Avg cumulative length
                elif filename in testing_epilength_all_trials and type == "testing_epilength":
                    data = testing_epilength_all_trials[filename]
                    heatmap_data[(reset_rate, epsilon)] = np.mean(np.cumsum(data, axis=1)[:, -1])  # Avg cumulative length
                elif filename in epilength_all_trials and type == "fpt":
                    data = epilength_all_trials[filename]
                    heatmap_data[(reset_rate, epsilon)] = np.mean(data[:, 0])  # Mean of the 0th element (first passage time)
                elif filename in epilength_all_trials and type == "mean_over_fpt":
                    data = epilength_all_trials[filename]
                    fpt = np.mean(data[:, 0])  # Mean of the 0th element (first passage time)
                    mean_cumulative_length = np.mean(np.cumsum(data, axis=1)[:, -1])  # Mean cumulative length, same as what was computed above
                    heatmap_data[(reset_rate, epsilon)] = mean_cumulative_length / fpt
                elif filename in training_done_epi_sum and type == "training_done_epi_sum":
                    data = training_done_epi_sum[filename]
                    heatmap_data[(reset_rate, epsilon)] = data / 1000  # Average of training_done_epi_sum over 1000 trials
                elif type == "max_index_std_mean":
                    fig, ax = plt.subplots()  # Create a temporary figure and axis
                    max_indices = plot_std_over_mean(ax, [filename], "epilength", ylabel, logscale=True, cumsum=False, plot_maxima=True)
                    plt.close(fig)  # Close the figure to suppress plotting
                    heatmap_data[(reset_rate, epsilon)] = max_indices.get(filename, np.nan) + 1  # Add 1 to convert to 1-based indexing
                else:
                    continue

                epsilons.add(epsilon)
                reset_rates.add(reset_rate)

        except Exception as e:
            print(f"Warning: Error processing {filename}: {str(e)}, skipping...")
            continue

    if not heatmap_data:
        print(f"Warning: No valid data for {type} heatmap")
        return

    # Sort reset rates and epsilons for consistent ordering
    reset_rates = sorted(reset_rates)
    epsilons = sorted(epsilons)

    # Create a 2D array for the heatmap
    heatmap_array = np.zeros((len(reset_rates), len(epsilons)))
    for i, reset_rate in enumerate(reset_rates):
        for j, epsilon in enumerate(epsilons):
            heatmap_array[i, j] = heatmap_data.get((reset_rate, epsilon), np.nan)

    # Apply Gaussian blur AFTER forming the heatmap array
    heatmap_array = gaussian_filter(heatmap_array, sigma=1)

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(heatmap_array, aspect='auto', cmap='viridis', origin='lower')

    # Add contour lines
    X, Y = np.meshgrid(range(len(epsilons)), range(len(reset_rates)))
    contour = ax.contour(heatmap_array, levels=10, colors='white', linewidths=1.0, origin='lower', extent=[0, len(epsilons), 0, len(reset_rates)])
    ax.clabel(contour, inline=True, fontsize=10, fmt="%d")

    # Add colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label(ylabel, fontsize=10)

    if type == "fpt":
        fig, ax = plt.subplots(figsize=(10, 6))
        mean_fpt_vs_r = []
        for reset_rate in reset_rates:
            fpts = [heatmap_data[(reset_rate, epsilon)] for epsilon in epsilons if (reset_rate, epsilon) in heatmap_data]
            if fpts:
                mean_fpt_vs_r.append((reset_rate, np.mean(fpts)))
        if mean_fpt_vs_r:
            rs, mean_fpts = zip(*mean_fpt_vs_r)
            ax.plot(rs, mean_fpts, color='red', marker='o', linestyle='--', label="Mean FPT vs Reset Rate")
            ax.set_xlabel("Reset Rate", fontsize=10)
            ax.set_ylabel("Mean First Passage Time", fontsize=10)
            ax.set_title("Mean FPT vs Reset Rate", fontsize=12)
            ax.legend(fontsize=8, loc='upper right')
            base_filename = get_base_filename(filenames[0])
            plt.savefig(f"figs/mean_fpt_vs_r_{base_filename}.png", bbox_inches='tight')

    if type == "epilength":
        min_indices = []
        for col in heatmap_array.T:
            if np.all(np.isnan(col)):
                min_indices.append(np.nan)
            else:
                min_indices.append(np.nanargmin(col))
        min_indices = np.array(min_indices)

        # Smooth the min_indices using a simple moving average with Gaussian filtering
        smoothed_min_indices = gaussian_filter(min_indices.astype(float), sigma=1)
        ax.plot(range(len(epsilons)), smoothed_min_indices, color='orange', marker='o', linestyle='--', label="Smoothed Min per Epsilon")
        ax.legend(fontsize=8, loc='upper right')

    # Set axis labels and ticks
    ax.set_xticks(range(len(epsilons)))
    ax.set_yticks(range(len(reset_rates)))
    ax.set_xticklabels([f"{epsilon:.5f}" for epsilon in epsilons], rotation=45, fontsize=8)
    ax.set_yticklabels([f"{reset_rate:.5f}" for reset_rate in reset_rates], fontsize=8)
    ax.set_xlabel("Epsilon", fontsize=10)
    ax.set_ylabel("Reset Rate", fontsize=10)
    ax.set_title(f"Heatmap of {ylabel}", fontsize=12)

    # Save the heatmap
    base_filename = get_base_filename(filenames[0])
    plt.savefig(f"figs/heatmap_{type}_{base_filename}.png", bbox_inches='tight')
    plt.show()

def plot_intersection_vs_epsilon(filenames, ylabel_epi, ylabel_steps, data_type):
    intersection_data_epi = {}
    intersection_data_steps = {}
    intersection_distances = {}
    epsilons = set()

    # Group data by epsilon
    epsilon_data_dict = {}

    for filename in filenames:
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found, skipping...")
            continue

        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]

        reset_rate, epsilon, _, _ = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)
        epsilons.add(epsilon)

        if epsilon not in epsilon_data_dict:
            epsilon_data_dict[epsilon] = {}

        data = epilength_all_trials.get(filename)
        if data is None:
            print(f"Warning: Data for {filename} not found, skipping...")
            continue

        if data_type in {"mean", "median", "max", "std"}:
            if data_type == "mean":
                epsilon_data_dict[epsilon][filename] = np.mean(data, axis=0)
            elif data_type == "median":
                epsilon_data_dict[epsilon][filename] = np.median(data, axis=0)
            elif data_type == "max":
                epsilon_data_dict[epsilon][filename] = np.max(data, axis=0)
            elif data_type == "std":
                epsilon_data_dict[epsilon][filename] = np.std(data, axis=0)
        elif data_type in {"mean_minus_median", "mean_over_median"}:
            # Store raw data so we can compute both mean and median later
            epsilon_data_dict[epsilon][filename] = data

    # Sort epsilons for consistent ordering
    epsilons = sorted(epsilons)

    for epsilon in epsilons:
        data_dict = epsilon_data_dict[epsilon]
        try:
            if data_type == "mean_minus_median":
                mean_data_dict = {filename: np.mean(data, axis=0) for filename, data in data_dict.items()}
                median_data_dict = {filename: np.median(data, axis=0) for filename, data in data_dict.items()}

                mean_intersection_index, mean_intersection_dist = find_intersection_point(mean_data_dict)
                median_intersection_index, median_intesect_dist = find_intersection_point(median_data_dict)

                if mean_intersection_index != -1 and median_intersection_index != -1:
                    intersection_data_epi[epsilon] = (mean_intersection_index + 1) - (median_intersection_index + 1)
                    intersection_data_steps[epsilon] = np.mean(
                        [mean_data[mean_intersection_index] - median_data[median_intersection_index]
                         for mean_data, median_data in zip(mean_data_dict.values(), median_data_dict.values())]
                    )
                else:
                    intersection_data_epi[epsilon] = np.nan
                    intersection_data_steps[epsilon] = np.nan
            elif data_type == "mean_over_median":
                mean_data_dict = {filename: np.mean(data, axis=0) for filename, data in data_dict.items()}
                median_data_dict = {filename: np.median(data, axis=0) for filename, data in data_dict.items()}

                mean_intersection_index, mean_intersection_dist = find_intersection_point(mean_data_dict)
                median_intersection_index, mean_intersection_dist = find_intersection_point(median_data_dict)

                if mean_intersection_index != -1 and median_intersection_index != -1:
                    intersection_data_epi[epsilon] = (mean_intersection_index + 1) / (median_intersection_index + 1)
                    intersection_data_steps[epsilon] = np.mean(
                        [mean_data[mean_intersection_index] / median_data[median_intersection_index]
                         for mean_data, median_data in zip(mean_data_dict.values(), median_data_dict.values())]
                    )
                else:
                    intersection_data_epi[epsilon] = np.nan
                    intersection_data_steps[epsilon] = np.nan
            else:
                intersection_index, intersection_dist = find_intersection_point(data_dict)
                intersection_data_epi[epsilon] = intersection_index + 1
                intersection_distances[epsilon] = intersection_dist
                if data_type == "mean":
                    intersection_data_steps[epsilon] = np.mean(
                        [data[intersection_index] for data in data_dict.values()]
                    )
                elif data_type == "median":
                    intersection_data_steps[epsilon] = np.median(
                        [data[intersection_index] for data in data_dict.values()]
                    )
                elif data_type == "max":
                    intersection_data_steps[epsilon] = np.max(
                        [data[intersection_index] for data in data_dict.values()]
                    )
                elif data_type == "std":
                    intersection_data_steps[epsilon] = np.std(
                        [data[intersection_index] for data in data_dict.values()]
                    )
        except ValueError:
            intersection_data_epi[epsilon] = np.nan
            intersection_data_steps[epsilon] = np.nan
            intersection_distances[epsilon] = np.nan

    # Extract data for plotting
    intersection_episodes = [intersection_data_epi[epsilon] for epsilon in epsilons]
    intersection_steps = [intersection_data_steps[epsilon] for epsilon in epsilons]
    avg_intersection_distances = [intersection_distances[epsilon] for epsilon in epsilons]

    # Smooth the data using a simple moving average
    smoothing_window = 3
    padded_episodes = np.pad(intersection_episodes, (smoothing_window // 2,), mode='edge')
    smoothed_episodes = np.convolve(padded_episodes, np.ones(smoothing_window) / smoothing_window, mode='valid')

    padded_steps = np.pad(intersection_steps, (smoothing_window // 2,), mode='edge')
    smoothed_steps = np.convolve(padded_steps, np.ones(smoothing_window) / smoothing_window, mode='valid')

    padded_distances = np.pad(avg_intersection_distances, (smoothing_window // 2,), mode='edge')
    smoothed_distances = np.convolve(padded_distances, np.ones(smoothing_window) / smoothing_window, mode='valid')

    # Plot the data
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.scatter(epsilons, intersection_episodes, color='blue', marker='s', label=ylabel_epi)
    ax1.plot(epsilons, smoothed_episodes, 'b--', label=f"Smoothed {ylabel_epi}")
    ax1.set_xlabel("Epsilon", fontsize=10)
    ax1.set_ylabel(ylabel_epi, color='b', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    ax2.scatter(epsilons, intersection_steps, color='red', marker='o', label=ylabel_steps)
    ax2.plot(epsilons, smoothed_steps, 'r--', label=f"Smoothed {ylabel_steps}")
    ax2.set_ylabel(ylabel_steps, color='r', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='r')

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis
    ax3.scatter(epsilons, avg_intersection_distances, color='green', marker='^', label="Avg Intersection Distance")
    ax3.plot(epsilons, smoothed_distances, 'g--', label="Smoothed Avg Intersection Distance")
    ax3.set_ylabel("Avg Intersection Distance", color='g', fontsize=10)
    ax3.tick_params(axis='y', labelcolor='g')

    base_filename = get_base_filename(filenames[0])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    fig.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')

    ax1.set_title(f"Intersection Episodes, Steps, and Distances vs Epsilon ({data_type.capitalize()} Data)", fontsize=12)

    plt.savefig(f"figs/intersection_{data_type}_vs_epsilon_{base_filename}.png", bbox_inches='tight')
    plt.show()

# TODO: implement code that plots a heatmap of learning episodes
# TODO: this function only plots a 1D heatmap but i'm currently absolutely done with debugging it
def plot_intersection_proportion_heatmap(filenames):
    intersection_proportions = {}
    epsilons = set()

    # Group data by epsilon
    epsilon_data_dict = {}

    for filename in filenames:
        if filename.endswith('.npz'):
            filename_stripped = filename[:-4]

        reset_rate, epsilon, _, _ = extract_resetting_rate_epsilon_learning_rate_gamma(filename_stripped)
        epsilons.add(epsilon)

        if epsilon not in epsilon_data_dict:
            epsilon_data_dict[epsilon] = {}

        data = epilength_all_trials[filename]
        epsilon_data_dict[epsilon][filename] = np.mean(data, axis=0)

    # Sort epsilons for consistent ordering
    epsilons = sorted(epsilons)

    # Create a heatmap for each reset rate
    reset_rates = sorted(set(
        extract_resetting_rate_epsilon_learning_rate_gamma(f[:-4])[0] for f in filenames if f.endswith('.npz')
    ))

    for reset_rate in reset_rates:
        intersection_proportions = {}

        for epsilon, data_dict in epsilon_data_dict.items():
            filtered_data_dict = {
                filename: data for filename, data in data_dict.items()
                if extract_resetting_rate_epsilon_learning_rate_gamma(filename[:-4])[0] == reset_rate
            }

            if not filtered_data_dict:
                intersection_proportions[epsilon] = np.nan
                continue

            try:
                # Find intersection point using the same method as the first function
                intersection_index, intersection_dist = find_intersection_point(filtered_data_dict)

                # Compute the cumulative step proportion up to the intersection episode
                avg_cum_steps_at_intersection = np.mean(
                    [np.sum(data[:intersection_index + 1]) for data in filtered_data_dict.values()]
                )
                avg_total_cum_steps = np.mean(
                    [np.sum(data) for data in filtered_data_dict.values()]
                )

                # Compute the proportion
                intersection_proportions[epsilon] = (
                    avg_cum_steps_at_intersection / avg_total_cum_steps
                    if avg_total_cum_steps > 0 else np.nan
                )
            except ValueError:
                # No intersection found
                intersection_proportions[epsilon] = np.nan

        # Create the heatmap array (1D since we are only plotting against epsilon)
        proportion_heatmap = np.array([intersection_proportions.get(epsilon, np.nan) for epsilon in epsilons])

        # Plot heatmap
        plt.figure(figsize=(12, 6))
        cax = plt.imshow(proportion_heatmap[np.newaxis, :], aspect='auto', cmap='viridis',
                         origin='lower', vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(cax)
        cbar.set_label("Proportion of Cumulative Steps at Intersection", fontsize=12)

        # Set axis labels and ticks
        plt.xticks(range(len(epsilons)), [f"{epsilon:.3f}" for epsilon in epsilons], rotation=45, fontsize=9)
        plt.yticks([], [])  # No y-axis ticks since it's a 1D heatmap
        plt.xlabel("Epsilon", fontsize=12)
        plt.title(f"Heatmap of Proportion of Cumulative Steps at Intersection Point\nReset Rate: {reset_rate:.5f}", fontsize=14)

        # Save the plot
        base_filename = get_base_filename(filenames[0])
        plt.savefig(f"figs/intersection_proportion_heatmap_resetrate_{reset_rate:.5f}_{base_filename}.png", bbox_inches='tight')
        plt.show()

# Define a function that combines all intersection-related visualizations
def plot_all_intersection_analyses(filenames):
    # First, create the original plots using the existing functions
    # (these will be separate plots already)
    plot_intersection_vs_epsilon(filenames, "Intersection Episode", "Steps at Intersection", 'mean')
    
    # Now add our new plots
    plot_intersection_proportion_heatmap(filenames)
    
    print("All intersection analyses have been plotted as separate figures.")


# Call the heatmap / intersection functions
# plot_heatmap_avg(filenames, "Avg Cumulative Episode Length", 'epilength')
# plot_heatmap_avg(filenames, "Avg Cumulative Final Path Length", 'length')
# plot_heatmap_avg(filenames, "Avg Cumulative Testing Episode Length", 'testing_epilength')
# plot_heatmap_avg(filenames, "Avg First Passage Time", 'fpt')
# plot_heatmap_avg(filenames, "Avg Cumulative Episode Length / FPT", 'mean_over_fpt')
# plot_heatmap_avg(filenames, "Avg Training Done Episode", 'training_done_epi_sum')
# plot_heatmap_avg(filenames, "Max Index of Std/Mean", 'max_index_std_mean')

# plot_intersection_vs_epsilon(filenames, "Intersection Episode of Mean Data", "Intersection Steps", 'mean')
# plot_intersection_vs_epsilon(filenames, "Intersection Episode of Median Data", "Intersection Steps", 'median')
# plot_intersection_vs_epsilon(filenames, "Intersection Episode of Max Data", "Intersection Steps", 'max')
# plot_intersection_vs_epsilon(filenames, "Intersection Episode of Std Data", "Intersection Steps", 'std')
# plot_intersection_vs_epsilon(filenames, "Difference Between Intersection Episodes of Mean and Median Data", "Intersection Steps", 'mean_minus_median')
# plot_intersection_vs_epsilon(filenames, "Ratio of Intersection Episodes of Mean and Median Data", "Intersection Steps", 'mean_over_median')

# plot_all_intersection_analyses(filenames)  # does not work yet