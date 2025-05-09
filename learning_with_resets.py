import collections
from collections import deque
import random
import math
import os
import sys
import logging
import logging.config
import argparse # parse arguments from stdin in bash script
import gymnasium as gym
import gym_simplegrid
import numpy as np
from datetime import datetime as dt
# from gym_simplegrid.envs import SimpleGridEnv
from gymnasium.utils.save_video import save_video
# import custom environments
from simplegrid_with_resets.envs import SimpleGridEnv, SimpleGridEnvResets
from simplegrid_with_resets.dfs import find_path # depth-first search for constructing gridworld
from learning_algorithms.QLearn import QLearn
from learning_algorithms.TDLambda import TDLambda  # Import the TDLambda class

# no obstacles
obstacle_prob = 0. # prob that cell in box initialized as obstacle

# # large system length
# total_length = int(2e3) # is this large enough? 
# # total_length = 50 # to visualize

def create_array_2D(system_size, obstacle_prob, start_x, start_y, goal_x, goal_y):
    arr = np.random.choice([0, 1], size=(system_size, system_size), p=[1 - obstacle_prob, obstacle_prob])
    return [''.join(map(str, row)) for row in arr]

def create_array_1D(total_length, obstacle_prob, start_s, goal_s):
    # Initialize an N x N array filled with zeros
    arr = np.random.choice([0, 1], size=(1, total_length), p=[1 - obstacle_prob, obstacle_prob])

    # Convert the array to the string form
    return [''.join(map(str, row)) for row in arr]

# this no longer matters
# def set_reset_rate_and_epsilon(reset_decay, training_done, reset_rate, q):
#     if reset_decay == 'linear':
#         reset_rate = 0.0
#         q.epsilon = 0.0
#     elif reset_decay == 'twomodes' and training_done:
#         reset_rate = 0.0
#         q.epsilon = 0.0
#     elif reset_decay == 'none':
#         # Do not change reset_rate or epsilon
#         pass
#     return reset_rate, q.epsilon

def main():

    # parse command-line arguments
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
    parser.add_argument('--resetting_mode', type=str, required=True) # resetting mode -- position / memory
    parser.add_argument('--N', type=int, required=True) # system size
    parser.add_argument('--system_size', type=int, required=True)
    parser.add_argument('--learning_end_condition', type=str, required=True) # options: QStable (N end-of-episodes at which the relative Q-table values are fixed) and threshold (path length without exploration is < 1.1 times taxicab)
    parser.add_argument('--dimension', type=int, required=True) # options: 1 and 2
    parser.add_argument('--trial_num', type=int, required=True)
    parser.add_argument('--evaluate_full_training', action='store_true') # flag to evaluate full training
    parser.add_argument('--strategy', type=str, choices=['epsilon-greedy', 'softmax'], default='epsilon-greedy') # action selection strategy
    parser.add_argument('--rwd', type=int, required=True)  # Add reward argument

    args = parser.parse_args()
    reset_rate = args.reset_rate
    learning_rate = args.learning_rate
    gamma = args.gamma
    epsilon = args.epsilon
    num_episodes = args.num_episodes
    learning_end_condition = args.learning_end_condition
    trial_num = args.trial_num
    evaluate_full_training = args.evaluate_full_training
    strategy = args.strategy
    reward = args.rwd

    # parse render mode argument
    if args.render_mode == "None":
        render_mode_arg = None
        # render_mode_arg = 'human' # to visualize
    else:
        render_mode_arg = args.render_mode # all other options have str inputs

    # dimension options
    N = args.N # system size
    dim = args.dimension
    system_size = args.system_size

    if dim == 1:
        system_geom = 'SimpleLineReset'
        start_s = int(system_size//2 - N//2)
        goal_s = int(system_size//2 + N//2)
        taxicab_length = goal_s - start_s

    elif dim == 2:
        start_x = int(system_size // 3)
        start_y = int(system_size // 3)
        goal_x = int(2 * system_size // 3)
        goal_y = int(2 * system_size // 3)

        # unravel to state space
        start_s = start_x + start_y*system_size
        goal_s = goal_x + goal_y*system_size

        # print(start_x, start_y, goal_x, goal_y)

        # for 2D, options for system geometry
        system_geom = 'SimpleGridReset-v0'  

        # define optimal solution to later calculate regret
        taxicab_length = np.absolute(goal_x - start_x) + np.absolute(goal_y - start_y)

    rwd_opt = -1 * taxicab_length
    reset_decay = args.reset_decay
    n_stable = args.n_stable
    resetting_mode = args.resetting_mode
    
    total_epilength_vec = np.empty(num_episodes)
    total_length_vec = np.empty(num_episodes)
    total_regret_vec = np.empty(num_episodes)
    total_testing_epilength_vec = np.empty(num_episodes)

    # initialize reward, regret, epilength vector filenames before we modify the reset_rate, epsilon, etc
    total_epilength_vec_file = f"vectors/total_epilength_vec_trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dim}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_rwd_{reward}.npy"
    
    # distinguish between episode length and LENGTH, which resets every reset (i.e. it is the eventual path from the start to the goal that the agent finds)
    total_length_vec_file = f"vectors/total_length_vec_trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dim}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_rwd_{reward}.npy"
    total_regret_vec_file = f"vectors/total_regret_vec_trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dim}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_rwd_{reward}.npy"
    total_training_done_epi_file = f"vectors/training_done_epi_trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dim}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_rwd_{reward}.npy"
    ending_regret_file = f"vectors/ending_regret_file_trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dim}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_rwd_{reward}.npy"
    total_testing_epilength_vec_file = f"vectors/total_testing_epilength_vec_trialnum_{trial_num}_resetrate_{reset_rate}_size_{N}_dimension_{dim}_systemsize_{system_size}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}_rwd_{reward}.npy"

    if dim == 2:
        env = gym.make(
            system_geom, 
            obstacle_map=create_array_2D(system_size, obstacle_prob, start_x, start_y, goal_x, goal_y), 
            render_mode=render_mode_arg,
        )

    elif dim == 1:
        env = gym.make(
            system_geom, 
            obstacle_map=create_array_1D(system_size, obstacle_prob, start_s, goal_s), 
            render_mode=render_mode_arg,
        )
    
    actions = list(env.unwrapped.MOVES.keys()) # actions defined in environment, then fed into QLearn
    q = QLearn(actions, epsilon, learning_rate, gamma, strategy=strategy)  # Initialize with strategy
    training_done = False # whether, at the given episode, training has completed

    if learning_end_condition == "QStable":
        stable_window = deque(maxlen=n_stable)

    training_done_epi = -1 
    consecutive_success = 0  # Counter for consecutive episodes meeting the condition

    for n_epi in range(num_episodes):
        # First run a "testing episode" with epsilon = 0.0 and reset rate = 0.0, and evaluate epilength
        q.epsilon = 0.0  # Turn off exploration: static policy
        s, _ = env.reset(options={'start_loc': start_s, 'goal_loc': goal_s, 'reset_rate': 0.0})
        done = False
        testing_epilength_this_episode = 0
        QTable_direction_incorrect = 0

        while not done:
            a = q.chooseAction(s)  # Choose action without exploration
            s_prime, r, done, truncated, info = env.step(a)
            s = s_prime
            testing_epilength_this_episode += 1
            if dim == 2 and not q.QDirectional(s):  # Check QDirectional only for 2D systems
                QTable_direction_incorrect += 1

        total_testing_epilength_vec[n_epi] = testing_epilength_this_episode

        # Evaluate whether training success condition is met
        success = (
            learning_end_condition == "threshold"
            and testing_epilength_this_episode <= taxicab_length * 1.025
            and (dim != 2 or QTable_direction_incorrect <= 1)
        )

        if success:
            consecutive_success += 1
        else:
            consecutive_success = 0  # Reset on failure

        # Declare training done after 5 consecutive successes
        if consecutive_success >= 5:
            training_done = True
            training_done_epi = n_epi

        # if training done, then save testing_epilength as epilength, length, and testing epilength, and regret accordingly
        if training_done:
            total_epilength_vec[n_epi] = testing_epilength_this_episode
            total_length_vec[n_epi] = testing_epilength_this_episode
            regret_this_episode = rwd_opt - (-1 * testing_epilength_this_episode)
            total_regret_vec[n_epi] = regret_this_episode
        
        else: # if training not done, then run a training episode
            q.epsilon = epsilon  # Set exploration rate
            s, _ = env.reset(options={'start_loc': start_s, 'goal_loc': goal_s, 'reset_rate': reset_rate})
            done = False
            reward_this_episode = 0
            epilength_this_episode = 0
            length_this_episode = 0

            while not done:
                a = q.chooseAction(s)  # Choose action with exploration
                s_prime, r, done, truncated, info = env.step(a)
                reset_last_step = info['reset_last_step']

                if reset_last_step:
                    length_this_episode = 0  # Reset path length

                else:
                    q.learn(s, a, r, s_prime)  # Update Q-table
                    length_this_episode += 1

                s = s_prime
                epilength_this_episode += 1
                reward_this_episode -= 1  # Decrease reward with each step

                if done:
                    # print('epilength', epilength_this_episode)  # debug
                    total_epilength_vec[n_epi] = epilength_this_episode
                    total_length_vec[n_epi] = length_this_episode
                    regret_this_episode = rwd_opt - reward_this_episode
                    total_regret_vec[n_epi] = regret_this_episode

    # save stored vectors to feed into bash script, which then writes them to one CSV file
    np.save(total_epilength_vec_file, total_epilength_vec)
    np.save(total_length_vec_file, total_length_vec)
    np.save(total_regret_vec_file, total_regret_vec)
    np.save(total_training_done_epi_file, training_done_epi)
    np.save(total_testing_epilength_vec_file, total_testing_epilength_vec)
    np.save(ending_regret_file, regret_this_episode) # ending regret is the last regret_this_episode

    env.close()

if __name__ == '__main__':
    main()