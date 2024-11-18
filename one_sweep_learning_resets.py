import os
import numpy as np
import subprocess
from multiprocessing import Pool
import time
import argparse

# sweep over the following (can increase number of hyperparameters)
# reset_rates = np.linspace(0.039, 0.07, 10)
# reset_rates = np.linspace(0.0101, 0.02, 100)
# reset_rates = np.linspace(0.0201,0.05,300)
# learning_rates = [0.0005]
# gammas = [0.965]
# epsilons = [0.06]
# n_stables = [30]
# qlearn_after_resets = [True]
# reset_decay = "twomodes"

# argparse to parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--reset_rate', type=float, required=True)
parser.add_argument('--learning_rate', type=float, required=True) # alpha
parser.add_argument('--gamma', type=float, required=True)
parser.add_argument('--epsilon', type=float, required=True) # exploration vs exploitation
parser.add_argument('--num_episodes', type=int, default=100)
parser.add_argument('--render_mode', type=str, default=None)
# there is some issue with stdin bool arguments
parser.add_argument('--qlearn_after_resets',type=str, default='True')
parser.add_argument('--reset_decay', type=str, default='none')
parser.add_argument('--n_stable', type=int, default=20)

args = parser.parse_args()
reset_rate = args.reset_rate
learning_rate = args.learning_rate
gamma = args.gamma
epsilon = args.epsilon
num_episodes = args.num_episodes
qlearn_after_resets_value = args.qlearn_after_resets
reset_decay = args.reset_decay
n_stable_value = args.n_stable
render_mode = args.render_mode

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

# function to execute a single combination of parameters
def run_sweep(reset_rate, learning_rate, gamma, epsilon, n_stable_value, qlearn_after_resets_value):
    print(f"Running for reset_rate={reset_rate}, learning_rate={learning_rate}, gamma={gamma}, epsilon={epsilon}, qlearnreset={qlearn_after_resets_value}, resetdecay={reset_decay}, n_stable={n_stable_value}")
    
    output_file_avg = f"results/resetrate_{reset_rate}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_qlearnreset_{qlearn_after_resets_value}_resetdecay_{reset_decay}_numepisodes_{num_episodes}.csv"
    with open(output_file_avg, 'w') as f:
        pass  # wipe the average output file

    reward_sum = None
    epilength_sum = None
    regret_sum = None
    training_done_epi_sum = 0
    ending_regret_sum = 0

    for trial in range(1, N_trials + 1):
        output_file = f"results/resetrate_{reset_rate}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_qlearnreset_{qlearn_after_resets_value}_resetdecay_{reset_decay}_numepisodes_{num_episodes}_trial_{trial}.csv"
        with open(output_file, 'w') as f:
            pass  # wipe individual output file

        print(f"  Trial {trial}...")

        cmd = f"python learning_with_resets.py --reset_rate {reset_rate} --learning_rate {learning_rate} --gamma {gamma} --epsilon {epsilon} --n_stable {n_stable_value} --reset_decay {reset_decay} --num_episodes {num_episodes} --render_mode {render_mode} --qlearn_after_resets {qlearn_after_resets_value}"
        try:
            run_command(cmd)
        except RuntimeError as e:
            print(e)
            return

        # read vectors
        reward_vec_file = f"vectors/total_reward_vec_resetrate_{reset_rate}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_qlearnreset_{qlearn_after_resets_value}_resetdecay_{reset_decay}.npy"
        epilength_vec_file = f"vectors/total_epilength_vec_resetrate_{reset_rate}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_qlearnreset_{qlearn_after_resets_value}_resetdecay_{reset_decay}.npy"
        regret_vec_file = f"vectors/total_regret_vec_resetrate_{reset_rate}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_qlearnreset_{qlearn_after_resets_value}_resetdecay_{reset_decay}.npy"
        training_done_epi_file = f"vectors/training_done_epi_resetrate_{reset_rate}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_qlearnreset_{qlearn_after_resets_value}_resetdecay_{reset_decay}.npy"
        ending_regret_file = f"vectors/ending_regret_file_resetrate_{reset_rate}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_qlearnreset_{qlearn_after_resets_value}_resetdecay_{reset_decay}.npy"

        # reward_vec = np.load('total_reward_vec.npy', allow_pickle=True)
        # epilength_vec = np.load('total_epilength_vec.npy', allow_pickle=True)
        # regret_vec = np.load('total_regret_vec.npy', allow_pickle=True)
        # training_done_epi = np.load('training_done_epi.npy', allow_pickle=True)

        reward_vec = np.load(reward_vec_file, allow_pickle=True)
        epilength_vec = np.load(epilength_vec_file, allow_pickle=True)
        regret_vec = np.load(regret_vec_file, allow_pickle=True)
        training_done_epi = np.load(training_done_epi_file, allow_pickle=True)
        ending_regret = np.load(ending_regret_file, allow_pickle=True)

        # save to individual output file
        with open(output_file, 'w') as f:
            np.savetxt(f, [reward_vec, epilength_vec, regret_vec], delimiter=',')

        # accumulate values for averaging
        if reward_sum is None:
            reward_sum = np.zeros_like(reward_vec)
            epilength_sum = np.zeros_like(epilength_vec)
            regret_sum = np.zeros_like(regret_vec)

        reward_sum += reward_vec
        epilength_sum += epilength_vec
        regret_sum += regret_vec
        training_done_epi_sum += training_done_epi
        ending_regret_sum += ending_regret

    # calculate averages
    reward_avg = reward_sum / N_trials
    epilength_avg = epilength_sum / N_trials
    regret_avg = regret_sum / N_trials
    training_done_epi_avg = training_done_epi_sum / N_trials
    ending_regret_avg = ending_regret_sum / N_trials

    # write averages
    with open(output_file_avg, 'w') as f:
        np.savetxt(f, [reward_avg, epilength_avg, regret_avg], delimiter=',')

    # log total regret and training completion
    log_file = 'results/parameter_sweep_log.csv'
    total_regret_across_episodes = np.sum(regret_avg) # INTEGRATED regret over episodes
    with open(log_file, 'a') as f:
        f.write(f"{reset_rate},{learning_rate},{gamma},{epsilon},{n_stable_value},{qlearn_after_resets_value},{reset_decay},{total_regret_across_episodes},{training_done_epi_avg},{ending_regret_avg} \n")

    # call plotting script
    cmd = f"python plotting.py --filename {output_file_avg}"
    try:
        run_command(cmd)
    except RuntimeError as e:
        print(e)

run_sweep(reset_rate, learning_rate, gamma, epsilon, n_stable_value, qlearn_after_resets_value) # actually run the sweep

# # pool of parameters
# param_list = [(reset_rate, lr, gamma, epsilon, n_stable, qlearn_after_resets_value)
#               for reset_rate in reset_rates
#               for lr in learning_rates
#               for gamma in gammas
#               for epsilon in epsilons
#               for n_stable in n_stables
#               for qlearn_after_resets_value in qlearn_after_resets]

# # run in parallel
# if __name__ == "__main__":
#     with Pool() as pool:
#         pool.starmap(run_sweep, param_list)