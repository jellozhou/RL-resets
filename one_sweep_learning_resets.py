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
parser.add_argument('--boundary', type=str, default='fixed')
parser.add_argument('--dimension', type=int, required=True)

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
boundary_type = args.boundary
learning_end_condition = args.learning_end_condition
dim = args.dimension

# parameters to keep fixed
N_trials = 100 # number of trials to avg over

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
def run_sweep(reset_rate, N, learning_rate, gamma, epsilon, n_stable_value, learning_end_condition, boundary_type, dimension):
    print(f"Running for reset_rate={reset_rate}, size={N}, learning_rate={learning_rate}, gamma={gamma}, epsilon={epsilon}, resetdecay={reset_decay}, n_stable={n_stable_value}, learning_end_condition={learning_end_condition}")
    
    output_file_avg = f"results/resetrate_{reset_rate}_size_{N}_dimension_{dim}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_numepisodes_{num_episodes}.csv"
    with open(output_file_avg, 'w') as f:
        pass  # wipe the average output file

    reward_sum = None
    epilength_sum = None
    regret_sum = None
    max_length = 0 # to track across trials
    training_done_epi_sum = 0
    ending_regret_sum = 0

    for trial in range(1, N_trials + 1):
        output_file = f"results/resetrate_{reset_rate}_size_{N}_dimension_{dim}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_numepisodes_{num_episodes}_trial_{trial}.csv"
        with open(output_file, 'w') as f:
            pass  # wipe individual output file

        print(f"  Trial {trial}...")

        cmd = f"python learning_with_resets.py --dimension {dim} --reset_rate {reset_rate} --N {N} --boundary {boundary_type} --learning_rate {learning_rate} --gamma {gamma} --epsilon {epsilon} --n_stable {n_stable_value} --learning_end_condition {learning_end_condition} --reset_decay {reset_decay} --resetting_mode {resetting_mode} --num_episodes {num_episodes} --render_mode {render_mode}"
        try:
            run_command(cmd)
        except RuntimeError as e:
            print(e)
            return

        # read vectors
        reward_vec_file = f"vectors/total_reward_vec_resetrate_{reset_rate}_size_{N}_dimension_{dim}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
        epilength_vec_file = f"vectors/total_epilength_vec_resetrate_{reset_rate}_size_{N}_dimension_{dim}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
        length_vec_file = f"vectors/total_length_vec_resetrate_{reset_rate}_size_{N}_dimension_{dim}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
        regret_vec_file = f"vectors/total_regret_vec_resetrate_{reset_rate}_size_{N}_dimension_{dim}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
        training_done_epi_file = f"vectors/training_done_epi_resetrate_{reset_rate}_size_{N}_dimension_{dim}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
        ending_regret_file = f"vectors/ending_regret_file_resetrate_{reset_rate}_size_{N}_dimension_{dim}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"

        reward_vec = np.load(reward_vec_file, allow_pickle=True)
        epilength_vec = np.load(epilength_vec_file, allow_pickle=True)
        # print(epilength_vec) # debug
        length_vec = np.load(length_vec_file, allow_pickle=True)
        regret_vec = np.load(regret_vec_file, allow_pickle=True)
        training_done_epi = np.load(training_done_epi_file, allow_pickle=True)
        ending_regret = np.load(ending_regret_file, allow_pickle=True)

        # update max length from loaded vectors
        # then later save this max length!
        max_length = max(max_length, len(reward_vec), len(epilength_vec), len(regret_vec))

        # adjust length of vectors for later summation
        reward_vec = extend_vector(reward_vec, max_length)
        epilength_vec = extend_vector(epilength_vec, max_length)
        regret_vec = extend_vector(regret_vec, max_length)
        length_vec = extend_vector(length_vec, max_length)

        # save to individual output file
        with open(output_file, 'w') as f:
            np.savetxt(f, [reward_vec, epilength_vec, regret_vec], delimiter=',')

        # accumulate values for averaging
        # if this is the first trial, initialize to vector of zeros
        if reward_sum is None:
            reward_sum = np.zeros(max_length)
            epilength_sum = np.zeros(max_length)
            regret_sum = np.zeros(max_length)
            length_sum = np.zeros(max_length)

        # Extend vectors to the maximum length encountered so far
        reward_sum = extend_vector(reward_sum, len(reward_sum))
        epilength_sum = extend_vector(epilength_sum, len(epilength_sum))
        regret_sum = extend_vector(regret_sum, len(regret_sum))
        length_sum = extend_vector(length_sum, len(length_sum))

        # Add the extended vectors
        reward_sum += reward_vec
        epilength_sum += epilength_vec
        # print("sum", epilength_sum) # debug
        length_sum += length_vec
        regret_sum += regret_vec
        training_done_epi_sum += training_done_epi
        ending_regret_sum += ending_regret
        print("Total regrets in last episodes:", ending_regret_sum)

    # calculate averages
    reward_avg = reward_sum / N_trials
    epilength_avg = epilength_sum / N_trials # when N=1, this is a way to find the first passage time
    # print(epilength_avg) # debug
    length_avg = length_sum / N_trials # same as above comment
    regret_avg = regret_sum / N_trials
    training_done_epi_avg = training_done_epi_sum / N_trials
    ending_regret_avg = ending_regret_sum / N_trials

    # write averages
    with open(output_file_avg, 'w') as f:
        # save the averaged vectors to eventually plot!
        np.savetxt(f, [reward_avg, epilength_avg, length_avg, regret_avg], delimiter=',')

    # log total regret and training completion

    # normal sweep across resetting rates, with learning
    # log_file = 'results/parameter_sweep_log.csv'

    log_file = f'log/parameter_sweep_log_{boundary_type}.csv'

    if dim == 1:
        log_file = f'log/parameter_sweep_log_{boundary_type}_dimension_1.csv'

    total_regret_across_episodes = np.sum(regret_avg) # INTEGRATED regret over episodes
    total_finallength_across_episodes = np.sum(length_avg) # integrated final path length (another kind of regret ig)
    with open(log_file, 'a') as f:              
        f.write(f"{reset_rate},{N},{boundary_type},{learning_rate},{gamma},{epsilon},{n_stable_value},{reset_decay},{total_regret_across_episodes},{total_finallength_across_episodes},{training_done_epi_avg},{ending_regret_avg},{epilength_avg[0]},{length_avg[0]},{max_length}\n") # there is the indexing since it's a vector

    # call plotting script
    cmd = f"python plotting.py --filename {output_file_avg}"
    try:
        run_command(cmd)
    except RuntimeError as e:
        print(e)

run_sweep(reset_rate, N, learning_rate, gamma, epsilon, n_stable_value, learning_end_condition, boundary_type, dim) # actually run the sweep