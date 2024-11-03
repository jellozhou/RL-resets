#!/bin/bash

# there is currently some bug in the code and I haven't yet debugged it

# Sweep over the following reset rates
# reset_rates=(0.00 0.01 0.02 0.03 0.04 0.05)
reset_rates=(0.00)
# learning_rates=(0.0000 0.0002 0.0004 0.0006 0.0008)
learning_rates=(0.0000 0.0002)
# gammas=(0.97 0.975 0.98 0.985 0.99 0.995)
gammas=(0.97)
qlearn_after_resets=("True" "False") # Flag for whether to learn in the step after a reset

# fixed parameters, for now
num_episodes=150
N_trials=2
render_mode='None'

# Create results directory
mkdir -p results
mkdir -p figs

# loop over hyperparameter values
for qlearn_after_resets_value in "${qlearn_after_resets[@]}"; do
    for gamma in "${gammas[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
            for reset_rate in "${reset_rates[@]}"; do
                echo "Running for reset_rate=${reset_rate}, learning_rate=${learning_rate}, gamma=${gamma}, qlearnreset=${qlearn_after_resets_value}"
                output_file_avg="results/resetrate_${reset_rate}_learningrate_${learning_rate}_gamma_${gamma}_qlearnreset_${qlearn_after_resets_value}_numepisodes_${num_episodes}.csv"
                > "$output_file_avg"  # Wipe the average output file

                # initialize cumulative arrays
                reward_sum=()
                epilength_sum=()
                regret_sum=()

                # loop over trials, then store to average
                for trial in $(seq 1 $N_trials); do
                    # create output files to store all results
                    output_file="results/resetrate_${reset_rate}_qlearnreset_${qlearn_after_resets_value}_numepisodes_${num_episodes}_trial_${trial}.csv"
                    > "$output_file"  # Wipe individual output file

                    echo "  Trial ${trial}..."
                    # run python script & check for errors while doing so
                    if ! python learning_with_resets.py --reset_rate $reset_rate --learning_rate $learning_rate --gamma $gamma --num_episodes $num_episodes --render_mode $render_mode --qlearn_after_resets $qlearn_after_resets_value; then
                        echo "Error running the Python script for trial ${trial}. Exiting."
                        exit 1
                    fi

                    # read reward, episode length, and regret vectors
                    reward_vec_raw=$(python -c "import numpy as np; print(','.join(map(str, np.load('total_reward_vec.npy', allow_pickle=True))))")
                    epilength_vec_raw=$(python -c "import numpy as np; print(','.join(map(str, np.load('total_epilength_vec.npy', allow_pickle=True))))")
                    regret_vec_raw=$(python -c "import numpy as np; print(','.join(map(str, np.load('total_regret_vec.npy', allow_pickle=True))))")

                    # split the raw comma-separated values into arrays
                    IFS=',' read -r -a reward_vec <<< "$reward_vec_raw"
                    IFS=',' read -r -a epilength_vec <<< "$epilength_vec_raw"
                    IFS=',' read -r -a regret_vec <<< "$regret_vec_raw"

                    # write to the individual output file
                    printf "%s\n" "${reward_vec[*]}" >> "$output_file"
                    printf "%s\n" "${epilength_vec[*]}" >> "$output_file"
                    printf "%s\n" "${regret_vec[*]}" >> "$output_file"

                    # initialize cumulative arrays if this is the first trial
                    if [ ${#reward_sum[@]} -eq 0 ]; then
                        for ((i=0; i<${#reward_vec[@]}; i++)); do
                            reward_sum[$i]=0
                            epilength_sum[$i]=0
                            regret_sum[$i]=0
                        done
                    fi

                    # add values to cumulative sums for averaging later
                    for i in "${!reward_vec[@]}"; do
                        reward_sum[$i]=$(echo "${reward_sum[$i]} + ${reward_vec[$i]}" | bc)
                        epilength_sum[$i]=$(echo "${epilength_sum[$i]} + ${epilength_vec[$i]}" | bc)
                        regret_sum[$i]=$(echo "${regret_sum[$i]} + ${regret_vec[$i]}" | bc)
                    done
                done

                # average across trials!
                for i in "${!reward_sum[@]}"; do
                    if [ -n "${reward_sum[$i]}" ]; then
                        reward_avg=$(echo "${reward_sum[$i]} / $N_trials" | bc -l)

                        # write averages to the avg output file
                        # echo $reward_avg
                        printf "%s" >> "$output_file_avg"
                        echo "$reward_avg" >> "$output_file_avg"
                    fi
                    # printf "\n" >> "$output_file_avg"
                done

                for i in "${!epilength_sum[@]}"; do
                    if [ -n "${epilength_sum[$i]}" ]; then
                        epilength_avg=$(echo "${epilength_sum[$i]} / $N_trials" | bc -l)

                        # Write averages to the avg output file
                        printf "%s" >> "$output_file_avg"
                        echo "$epilength_avg" >> "$output_file_avg"
                    fi
                    # printf "\n" >> "$output_file_avg"
                done

                for i in "${!regret_sum[@]}"; do
                    if [ -n "${regret_sum[$i]}" ]; then
                        regret_avg=$(echo "${regret_sum[$i]} / $N_trials" | bc -l)

                        # Write averages to the avg output file
                        printf "%s" >> "$output_file_avg"
                        echo "$regret_avg" >> "$output_file_avg"
                    fi
                done

                python plotting.py --filename "$output_file_avg"
            done
        done
    done
done