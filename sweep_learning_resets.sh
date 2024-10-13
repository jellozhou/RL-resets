#!/bin/bash

# sweep over the following reset rates
reset_rates=(0.00 0.01 0.02 0.03 0.04 0.05)

num_episodes=100
N_trials=10
# render_mode='None'

# output file to store all results
timestamp=$(date +%Y%m%d_%H%M%S)
output_file_reset_rates="results/parameter_sweep_reset_rates_${timestamp}.csv"
output_file_rewards="results/parameter_sweep_rewards_${timestamp}.csv"
output_file_epilengths="results/parameter_sweep_epilengths_${timestamp}.csv"

# create results directory
mkdir -p results

# initialize the CSV file with headers
# nevermind
# echo "reset_rate,trial,total_reward_vec,total_epilength_vec" > $output_file

# loop over the reset_rate values
for reset_rate in "${reset_rates[@]}"; do
    echo "Running for reset_rate=${reset_rate}"

    # loop over trials
    for trial in $(seq 1 $N_trials); do
        echo "  Trial ${trial}..."
        python learning_with_resets.py --reset_rate $reset_rate --num_episodes $num_episodes

        # append results to initialized CSV files
        echo "${reset_rate}" >> $output_file_reset_rates
        # paste -d, $output_file_rewards $(echo )
        # echo "$(paste total_reward_vec.npy)" >> $output_file_rewards
        # echo "$(paste total_epilength_vec.npy)" >> $output_file_epilengths
        # # echo "${reset_rate},$(cat total_reward_vec.npy),$(cat total_epilength_vec.npy)" >> $output_file
        
        # append total rewards and episode lengths to their respective CSV files
        echo "$(python -c "import numpy as np; print(','.join(map(str, np.loadtxt('total_reward_vec.npy'))))")" >> $output_file_rewards
        echo "$(python -c "import numpy as np; print(','.join(map(str, np.loadtxt('total_epilength_vec.npy'))))")" >> $output_file_epilengths
    done
done

# make plots