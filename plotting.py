import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

# parse arguments from bash script
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default=None)
args = parser.parse_args()
filename = args.filename

# debug: specify a filename
# filename = "results/resetrate_0.01_qlearnreset_False_numepisodes_10.csv"

if filename.endswith('.csv'):
    filename_stripped = filename[:-4]  # Remove exactly 4 characters from the end
filename_stripped = filename_stripped.split('/')[-1]

# parse variables from filename, return a dict
def extract_variables(filename):
    parts = filename.split('_')   
    # print(parts)
    variables = {}

    # loop through the parts two at a time (variable and value)
    for i in range(0, len(parts) - 1, 2):
        variable = parts[i]
        value = parts[i + 1]      
        # convert variables to numeric values
        if re.match(r'^-?\d+(\.\d+)?$', value):
            value = float(value) if '.' in value else int(value)      
        # add the variable-value pair to the dictionary
        variables[variable] = value 
    return variables

avg_array = np.loadtxt(filename)
N_trials = int(avg_array.size / 3)

# divide into reward, episode length, regret arrays
avg_reward_arr = avg_array[:N_trials]
avg_epilength_arr = avg_array[N_trials:2*N_trials]
avg_regret_arr = avg_array[2*N_trials:]

# construct episode number array
episode_num = np.linspace(1, avg_reward_arr.size, avg_reward_arr.size)

plt.figure()
plt.plot(episode_num, avg_reward_arr)
plt.title(extract_variables(filename_stripped))
plt.xlabel("Episode number")
plt.ylabel("Average reward per episode")
plt.savefig("figs/avg_reward_"+filename_stripped+".png")
# plt.show()

plt.figure()
plt.plot(episode_num, avg_epilength_arr)
plt.title(extract_variables(filename_stripped))
plt.xlabel("Episode number")
plt.ylabel("Average length per episode")
plt.savefig("figs/avg_epilength_"+filename_stripped+".png")
# plt.show()

plt.figure()
plt.plot(episode_num, avg_regret_arr)
plt.title(extract_variables(filename_stripped))
plt.xlabel("Episode number")
plt.ylabel("Average regret per episode")
plt.savefig("figs/avg_regret_"+filename_stripped+".png")
# plt.show()