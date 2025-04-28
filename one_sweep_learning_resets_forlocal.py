import os
import numpy as np
import subprocess
from multiprocessing import Pool
import time
import argparse

# argparse to parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--reset_rate', type=float, required=True)
parser.add_argument('--learning_rate', type=float, required=True) # alpha
parser.add_argument('--gamma', type=float, required=True)
parser.add_argument('--epsilon', type=float, required=True) # exploration vs exploitation
parser.add_argument('--num_episodes', type=int, default=100)
parser.add_argument('--render_mode', type=str, default=None)
# there is some issue with stdin bool arguments
parser.add_argument('--reset_decay', type=str, default='none')
parser.add_argument('--n_stable', type=int, default=20)
parser.add_argument('--learning_end_condition', type=str, required=True)
parser.add_argument('--resetting_mode', type=str, required=True)
parser.add_argument('--N', type=int, required=True) # system size
parser.add_argument('--system_size', type=int, required=True)
parser.add_argument('--dimension', type=int, required=True)
parser.add_argument('--trial_num', type=int, required=True)
parser.add_argument('--evaluate_full_training', action='store_true') # flag to evaluate full training
parser.add_argument('--strategy', type=str, choices=['epsilon-greedy', 'softmax'], default='epsilon-greedy')  # Add strategy argument
parser.add_argument('--rwd', type=int, required=True)  # Add reward argument

args = parser.parse_args()
reset_rate = args.reset_rate
learning_rate = args.learning_rate
gamma = args.gamma
epsilon = args.epsilon
num_episodes = args.num_episodes
reset_decay = args.reset_decay
n_stable_value = args.n_stable
render_mode = args.render_mode
resetting_mode = args.resetting_mode
N = args.N
system_size = args.system_size
learning_end_condition = args.learning_end_condition
dim = args.dimension
trial_num = args.trial_num
evaluate_full_training = args.evaluate_full_training
strategy = args.strategy
reward = args.rwd

# parameters to keep fixed
N_trials = 1000 # number of trials to avg over

# create results directory
os.makedirs('results', exist_ok=True)
os.makedirs('figs', exist_ok=True)
os.makedirs('vectors', exist_ok=True) # regret, reward, episode length vector directory

# helper function to run a command in bash and capture errors
def run_command(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

# below: required for adjustable episode lengths. 
def extend_vector(vec, target_length):
    """Extend a vector to the target length by repeating the last value."""
    if len(vec) < target_length:
        vec = np.append(vec, [vec[-1]] * (target_length - len(vec)))
    return vec

# function to execute a single combination of parameters
def run_sweep(trial_num, reset_rate, N, learning_rate, gamma, epsilon, n_stable_value, learning_end_condition, system_size, dimension, reward):
    print(f"Running for reset_rate={reset_rate}, size={N}, learning_rate={learning_rate}, gamma={gamma}, epsilon={epsilon}, resetdecay={reset_decay}, n_stable={n_stable_value}, learning_end_condition={learning_end_condition}, reward={reward}")
    
    base_filename = f"results/trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dimension}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_rwd_{reward}_numepisodes_{num_episodes}"
    max_length = 1 # to track across trials; set to 1 and not 0 so vector[-1] works

    # initialize vectors, extend later
    epilength_all_trials = []
    regret_all_trials = []
    length_all_trials = []
    testing_epilength_all_trials = []

    training_done_epi_sum = 0
    ending_regret_sum = 0
    num_training_not_done = 0 # number of episodes in which training is not done

    for trial in range(1, N_trials + 1):
        output_file = f"results/trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dim}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_rwd_{reward}_numepisodes_{num_episodes}_trial_{trial}.csv"
        with open(output_file, 'w') as f:
            pass  # wipe individual output file

        cmd = f"python learning_with_resets.py --trial_num {trial_num} --dimension {dim} --reset_rate {reset_rate} --N {N} --system_size {system_size} --learning_rate {learning_rate} --gamma {gamma} --epsilon {epsilon} --n_stable {n_stable_value} --learning_end_condition {learning_end_condition} --reset_decay {reset_decay} --resetting_mode {resetting_mode} --num_episodes {num_episodes} --render_mode {render_mode} --strategy {strategy} --rwd {reward} {'--evaluate_full_training' if evaluate_full_training else ''}"
        try:
            run_command(cmd)
        except RuntimeError as e:
            print(e)
            return

        # read vectors
        epilength_vec_file = f"vectors/total_epilength_vec_trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dim}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_rwd_{reward}.npy"
        length_vec_file = f"vectors/total_length_vec_trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dim}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_rwd_{reward}.npy"
        regret_vec_file = f"vectors/total_regret_vec_trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dim}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_rwd_{reward}.npy"
        testing_epilength_vec_file = f"vectors/total_testing_epilength_vec_trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dim}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_rwd_{reward}.npy"
        training_done_epi_file = f"vectors/training_done_epi_trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dim}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_rwd_{reward}.npy"
        ending_regret_file = f"vectors/ending_regret_file_trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dim}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_rwd_{reward}.npy"

        epilength_vec = np.load(epilength_vec_file, allow_pickle=True)
        length_vec = np.load(length_vec_file, allow_pickle=True)
        regret_vec = np.load(regret_vec_file, allow_pickle=True)
        testing_epilength_vec = np.load(testing_epilength_vec_file, allow_pickle=True)
        training_done_epi = np.load(training_done_epi_file, allow_pickle=True)
        ending_regret = np.load(ending_regret_file, allow_pickle=True)
      
        # update max length from loaded vectors
        max_length = max(max_length, len(epilength_vec), len(regret_vec))

        # adjust length of vectors for later summation
        epilength_vec = extend_vector(epilength_vec, max_length)
        regret_vec = extend_vector(regret_vec, max_length)
        length_vec = extend_vector(length_vec, max_length)
        testing_epilength_vec = extend_vector(testing_epilength_vec, max_length)

        # save to individual output file
        with open(output_file, 'w') as f:
            np.savetxt(f, [epilength_vec, regret_vec, testing_epilength_vec], delimiter=',')

        if training_done_epi != -1:
            training_done_epi_sum += training_done_epi

        ending_regret_sum += ending_regret

        if training_done_epi == -1:
            num_training_not_done += 1

        # append new results to make list of vectors for all trials
        epilength_all_trials.append(epilength_vec)
        # print("epilength all trials", epilength_all_trials) # debug
        regret_all_trials.append(regret_vec)
        length_all_trials.append(length_vec)
        testing_epilength_all_trials.append(testing_epilength_vec)

    # calculate some averages
    training_done_epi_avg = training_done_epi_sum / N_trials
    ending_regret_avg = ending_regret_sum / N_trials

    # calculate averages, to input into log file
    epilength_avg = np.mean(np.array(epilength_all_trials), axis=0)
    length_avg = np.mean(np.array(length_all_trials), axis=0)
    regret_avg = np.mean(np.array(regret_all_trials), axis=0)

    # delete original files to save space
    files_to_delete = [
        epilength_vec_file, length_vec_file, regret_vec_file,
        training_done_epi_file, ending_regret_file, testing_epilength_vec_file
    ]
    
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)

    # Save all trial data into a single file
    output_npz_file = f"{base_filename}_full_data.npz"
    np.savez_compressed(
        output_npz_file,
        epilength_all_trials=epilength_all_trials,
        regret_all_trials=regret_all_trials,
        length_all_trials=length_all_trials,
        testing_epilength_all_trials=testing_epilength_all_trials,
        training_done_epi_sum=training_done_epi_sum,
        ending_regret_sum=ending_regret_sum,
        num_training_not_done=num_training_not_done,
        max_length=max_length
    )

    print(f"Saved full trial data to {output_npz_file}")

    log_file = f'log/test.csv'

    if dim == 1:
        log_file = f'log/parameter_sweep_log_dimension_1.csv'

    total_regret_across_episodes = np.sum(regret_avg)
    # print("debug: final regret avg", total_regret_across_episodes)
    total_finallength_across_episodes = np.sum(length_avg)
    with open(log_file, 'a') as f:              
        f.write(f"{reset_rate},{N},{system_size},{num_training_not_done},{learning_rate},{gamma},{epsilon},{n_stable_value},{reset_decay},{total_regret_across_episodes},{total_finallength_across_episodes},{training_done_epi_avg},{ending_regret_avg},{epilength_avg[0]},{length_avg[0]},{max_length}\n")

    # call plotting script
    # THIS is what causes the duplicate plots!
    # cmd = f"python plotting.py --filename {output_files['avg']}"
    # try:
    #     run_command(cmd)
    # except RuntimeError as e:
    #     print(e)

run_sweep(trial_num, reset_rate, N, learning_rate, gamma, epsilon, n_stable_value, learning_end_condition, system_size, dim, reward)