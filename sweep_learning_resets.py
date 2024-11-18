import os
import numpy as np
import subprocess

# sweep over the following (can increase number of hyperparameters)
reset_rates = np.linspace(0.039, 0.07, 10)
# reset_rates = [0.0375, 0.04, 0.04, 0.0425, 0.0425, 0.045, 0.045, 0.0475, 0.0475]  # first just fix it to be 0 for the hyperparameter sweep
# reset_rates = [0.03]
# learning_rates = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006]
learning_rates = [0.0005]
gammas = [0.965]  # optimize over learning rates first, then gamma
# epsilons = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
epsilons = [0.06]
n_stables = [30] # number of stable Q-tables to turn off reset
# qlearn_after_resets = [True, False]
qlearn_after_resets = [True]
reset_decay = "twomodes" # options: "none", "twomodes", "linear"

# parameters to keep fixed
num_episodes = 300
N_trials = 100
render_mode = 'None'

# create results directory
os.makedirs('results', exist_ok=True)
os.makedirs('figs', exist_ok=True)

# helper function to run a command in bash and capture errors
def run_command(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

# initialize or open a file to log parameter values and total regret across episodes
log_file = 'results/parameter_sweep_log.csv'
if not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        # write the header row if the file doesn't exist
        f.write("reset_rate,learning_rate,gamma,qlearn_after_resets,total_regret_across_episodes\n")

# loop over hyperparameter values
for n_stable_value in n_stables:
    for qlearn_after_resets_value in qlearn_after_resets:
        for epsilon in epsilons: 
            for gamma in gammas:
                for learning_rate in learning_rates:
                    for reset_rate in reset_rates:
                        print(f"Running for reset_rate={reset_rate}, learning_rate={learning_rate}, gamma={gamma}, epsilon={epsilon}, qlearnreset={qlearn_after_resets_value}, resetdecay={reset_decay}, n_stable={n_stable_value}")
                        
                        output_file_avg = f"results/resetrate_{reset_rate}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_qlearnreset_{qlearn_after_resets_value}_resetdecay_{reset_decay}_numepisodes_{num_episodes}.csv"
                        with open(output_file_avg, 'w') as f:
                            pass  # wipe the average output file
                        
                        # initialize cumulative arrays
                        reward_sum = None
                        epilength_sum = None
                        regret_sum = None
                        
                        # loop over trials
                        for trial in range(1, N_trials + 1):
                            output_file = f"results/resetrate_{reset_rate}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable_value}_qlearnreset_{qlearn_after_resets_value}_resetdecay_{reset_decay}_numepisodes_{num_episodes}_trial_{trial}.csv"
                            with open(output_file, 'w') as f:
                                pass  # wipe individual output file

                            print(f"  Trial {trial}...")

                            # run python script & check for errors
                            cmd = f"python learning_with_resets.py --reset_rate {reset_rate} --learning_rate {learning_rate} --gamma {gamma} --epsilon {epsilon} --n_stable {n_stable_value} --reset_decay {reset_decay} --num_episodes {num_episodes} --render_mode {render_mode} --qlearn_after_resets {qlearn_after_resets_value}"
                            try:
                                run_command(cmd)
                            except RuntimeError as e:
                                print(e)
                                print(f"Error running the Python script for trial {trial}. Exiting.")
                                exit(1)

                            # read reward, episode length, and regret vectors, and training_done_epi episode num
                            reward_vec = np.load('total_reward_vec.npy', allow_pickle=True)
                            epilength_vec = np.load('total_epilength_vec.npy', allow_pickle=True)
                            regret_vec = np.load('total_regret_vec.npy', allow_pickle=True)
                            training_done_epi = np.load('training_done_epi.npy', allow_pickle=True)

                            if os.path.exists('training_done_epi.npy'):
                                training_done_epi = np.load('training_done_epi.npy', allow_pickle=True)
                                print(training_done_epi)
                            else:
                                print("training_done_epi.npy does not exist!")
                            # write to the individual output file
                            with open(output_file, 'w') as f:
                                np.savetxt(f, [reward_vec, epilength_vec, regret_vec], delimiter=',')

                            # initialize cumulative arrays if this is the first trial
                            if reward_sum is None:
                                reward_sum = np.zeros_like(reward_vec)
                                epilength_sum = np.zeros_like(epilength_vec)
                                regret_sum = np.zeros_like(regret_vec)
                                training_done_epi_sum = 0

                            # add values to cumulative sums for averaging later
                            reward_sum += reward_vec
                            epilength_sum += epilength_vec
                            regret_sum += regret_vec
                            training_done_epi_sum += training_done_epi

                        # average across trials
                        reward_avg = reward_sum / N_trials
                        epilength_avg = epilength_sum / N_trials
                        regret_avg = regret_sum / N_trials
                        training_done_epi_avg = training_done_epi_sum / N_trials

                        # write averages to the avg output file
                        with open(output_file_avg, 'w') as f:
                            np.savetxt(f, [reward_avg, epilength_avg, regret_avg], delimiter=',')

                        # save total integrated regret across episodes, for rough optimization of each hyperparameter
                        total_regret_across_episodes = np.sum(regret_vec)
                        # append parameter values and total regret to the log file
                        # also append the average episode number when training is done
                        with open(log_file, 'a') as f:
                            f.write(f"{reset_rate},{learning_rate},{gamma},{epsilon},{n_stable_value},{qlearn_after_resets_value},{reset_decay},{total_regret_across_episodes},{training_done_epi_avg} \n")

                        # call plotting script
                        cmd = f"python plotting.py --filename {output_file_avg}"
                        try:
                            run_command(cmd)
                        except RuntimeError as e:
                            print(e)
                            print(f"Error running the plotting script.")