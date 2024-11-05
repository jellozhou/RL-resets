import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

# packages to loop over a directory and generate plots
import os
import fnmatch

# parse variables from filename, return a dict
def extract_variables(filename):
    parts = filename.split('_')   
    variables = {}
    # loop through the parts two at a time (variable and value)
    for i in range(0, len(parts) - 1, 2):
        variable = parts[i]
        value = parts[i + 1]      
        # convert variables to numeric values
        if re.match(r'^-?\d+(\.\d+)?$', value):
            value = float(value) if '.' in value else int(value)
        # exclude 'numepisodes' from being added to the dictionary cuz it's redundant
        if variable.lower() != "numepisodes":
            variables[variable] = value        
    return variables

# title formatting
def format_title(variables):
    # create a string representation and join them with newlines for better formatting
    title_str = ", ".join([f"{key}: {value}" for key, value in variables.items()])
    return title_str

# parse arguments from bash script
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default=None)
args = parser.parse_args()
filename = args.filename

# ----- code to loop over a directory and generate plots -----
# directory = 'results/newest_exp/avgfile'
# for filenameraw in os.listdir(directory):
#     if '_trial' not in filenameraw and 'parameter' not in filenameraw and filenameraw.endswith('.csv'):
#         filename = os.path.join(directory, filenameraw)
#         print(filename)

if filename.endswith('.csv'):
    filename_stripped = filename[:-4]  # Remove exactly 4 characters from the end
filename_stripped = filename_stripped.split('/')[-1]

avg_array = np.loadtxt(filename, delimiter=',')
N_trials = int(avg_array.size / 3)

# divide into reward, episode length, regret arrays
avg_reward_arr = avg_array[0]
avg_epilength_arr = avg_array[1]
avg_regret_arr = avg_array[2]

# construct episode number array
episode_num = np.linspace(1, avg_reward_arr.size, avg_reward_arr.size)

plt.figure()
plt.plot(episode_num, avg_reward_arr)
plt.title(format_title(extract_variables(filename_stripped)), fontsize=8)
plt.xlabel("Episode number")
plt.ylabel("Average reward per episode")
plt.savefig("figs/avg_reward_"+filename_stripped+".png")
# plt.show()

plt.figure()
plt.plot(episode_num, avg_epilength_arr)
plt.title(format_title(extract_variables(filename_stripped)), fontsize=8)
plt.xlabel("Episode number")
plt.ylabel("Average length per episode")
plt.savefig("figs/avg_epilength_"+filename_stripped+".png")
# plt.show()

plt.figure()
plt.plot(episode_num, avg_epilength_arr)
plt.title(format_title(extract_variables(filename_stripped)), fontsize=8)
plt.xlabel("Episode number")
plt.ylabel("Average length per episode")
plt.xscale('log')
plt.yscale('log')
plt.savefig("figs/logscale/loglog_avg_epilength_"+filename_stripped+".png")
# plt.show()

plt.figure()
plt.plot(episode_num, avg_regret_arr)
plt.title(format_title(extract_variables(filename_stripped)), fontsize=8)
plt.xlabel("Episode number")
plt.ylabel("Average regret per episode")
plt.savefig("figs/avg_regret_"+filename_stripped+".png")
# plt.show()

# loglog plot of regret
plt.figure()
plt.plot(episode_num, avg_regret_arr)
plt.yscale('log')
plt.xscale('log')
plt.title(format_title(extract_variables(filename_stripped)), fontsize=8)
plt.xlabel("Episode number")
plt.ylabel("Average regret per episode")
plt.savefig("figs/logscale/loglog_avg_regret_"+filename_stripped+".png")
# plt.show()