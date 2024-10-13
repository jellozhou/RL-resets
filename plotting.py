# doesn't work yet

import matplotlib.pyplot as plt
import numpy as np
import argparse

# parse arguments from bash script
parser = argparse.ArgumentParser()
parser.add_argument('--num_episodes', type=int, default=100)
args = parser.parse_args()
num_episodes = args.num_episodes

episodes_array = np.linspace(1, num_episodes, num_episodes)
reset_rate_array = np.loadtxt('results/parameter_sweep_reset_rates_20241012_212243.csv')
total_epilengths_array = np.loadtxt('results/parameter_sweep_epilengths_20241012_212243.csv', delimiter=',')
total_rewards_array = np.loadtxt('results/parameter_sweep_rewards_20241012_212243.csv', delimiter=',')
# print(reset_rate_array.shape)
# print(total_epilengths_array.shape)

# average the total arrays over each unique resetting rate
unique_rates, indices = np.unique(reset_rate_array, return_inverse=True)
avg_epilengths_array = np.empty_like(episodes_array)
avg_rewards_array = np.empty_like(episodes_array)

for i, rate in enumerate(unique_rates):
    avg_epilengths_array[i] = np.mean(total_epilengths_array[indices == i,:])
    print(avg_epilengths_array.shape)
    avg_rewards_array[i] = np.mean(total_rewards_array[indices == i,:])

plt.figure()
plt.plot(episodes_array, avg_epilengths_array)
plt.xlabel("Episode number")
plt.ylabel("Average episode step length")
plt.savefig("figs/episode_length_versus_number.png")

plt.figure()
plt.plot(episodes_array, avg_rewards_array)
plt.xlabel("Episode number")
plt.ylabel("Total reward per episode")
plt.savefig("figs/episode_reward_versus_number.png")