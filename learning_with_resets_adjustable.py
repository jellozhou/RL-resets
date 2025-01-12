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

# the same as learning_with_resets.py, but adjusts the number of training episodes dynamically. 

# no obstacles
obstacle_prob = 0. # prob that cell in box initialized as obstacle

def create_array(N, obstacle_prob, start_x, start_y, goal_x, goal_y):
    # Initialize an N x N array filled with zeros
    arr = np.random.choice([0, 1], size=(N, N), p=[1 - obstacle_prob, obstacle_prob])

    # iterate until there is a path from start to goal
    # this returns a "maximum recursion depth exceeded" error for large system sizes, and is also unnecessary in our setup
    # while find_path(arr, (start_x, start_y), (goal_x, goal_y)) == False:
    #     arr = np.random.choice([0, 1], size=(N, N), p=[1 - obstacle_prob, obstacle_prob])

    # Convert the array to the string form
    return [''.join(map(str, row)) for row in arr]

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
    parser.add_argument('--boundary', type=str, default='fixed')
    parser.add_argument('--learning_end_condition', type=str, required=True) # options: QStable (N end-of-episodes at which the relative Q-table values are fixed) and threshold (path length without exploration is < 1.1 times taxicab)

    args = parser.parse_args()
    reset_rate = args.reset_rate
    learning_rate = args.learning_rate
    gamma = args.gamma
    epsilon = args.epsilon
    # num_episodes = args.num_episodes # useless here
    learning_end_condition = args.learning_end_condition

    # parse render mode argument
    if args.render_mode == "None":
        render_mode_arg = None
        # render_mode_arg = 'human' # to visualize
    else:
        render_mode_arg = args.render_mode # all other options have str inputs

    # options for system geometry
    boundary_type = args.boundary
    if boundary_type == 'fixed':
        system_geom = 'SimpleGridReset-v0'
    elif boundary_type == 'periodic':
        system_geom = 'SimpleGridResetPBC-v0'

    reset_decay = args.reset_decay
    n_stable = args.n_stable
    resetting_mode = args.resetting_mode

    N = args.N # system size
    # convert starting & goal (x,y) to state
    start_x = int(N//3)
    start_y = int(N//3)
    goal_x = int(2*N//3)
    goal_y = int(2*N//3)
    start_s = start_x + start_y*N
    goal_s = goal_x + goal_y*N

    # define optimal solution to later calculate regret
    taxicab_length = np.absolute(goal_x - start_x) + np.absolute(goal_y - start_y)
    rwd_opt = -1 * taxicab_length
    
    total_reward_vec = np.empty(0)
    total_epilength_vec = np.empty(0)
    total_length_vec = np.empty(0)
    total_regret_vec = np.empty(0)

    # initialize reward, regret, epilength vector filenames before we modify the reset_rate, epsilon, etc
    total_reward_vec_file = f"vectors/total_reward_vec_resetrate_{reset_rate}_size_{N}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
    total_epilength_vec_file = f"vectors/total_epilength_vec_resetrate_{reset_rate}_size_{N}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
    
    # distinguish between episode length and LENGTH, which resets every reset (i.e. it is the eventual path from the start to the goal that the agent finds)
    total_length_vec_file = f"vectors/total_length_vec_resetrate_{reset_rate}_size_{N}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
    total_regret_vec_file = f"vectors/total_regret_vec_resetrate_{reset_rate}_size_{N}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
    total_training_done_epi_file = f"vectors/training_done_epi_resetrate_{reset_rate}_size_{N}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
    ending_regret_file = f"vectors/ending_regret_file_resetrate_{reset_rate}_size_{N}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_learningend_{learning_end_condition}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"

    env = gym.make(
        system_geom, 
        obstacle_map=create_array(N, obstacle_prob, start_x, start_y, goal_x, goal_y), 
        render_mode=render_mode_arg
    )
    
    actions = list(env.unwrapped.MOVES.keys()) # actions defined in environment, then fed into QLearn
    q = QLearn(actions, epsilon, learning_rate, gamma) # initialize
    training_done = False # whether, at the given episode, training has completed

    if learning_end_condition == "QStable":
        stable_window = deque(maxlen=n_stable)

    training_done_epi = -1 # what to do if training never done?? 
    n_epi = 0 
    # for n_epi in range(num_episodes):
    while training_done == False: # before training is completed -- afterwards don't run to save computational power! 
        print("Current episode number: ", n_epi)
        # initialize reward, episode length, regret
        reward_this_episode = 0
        epilength_this_episode = 0
        length_this_episode = 0

        # set epsilon values according to the decay option
        q.epsilon = epsilon # no annealing yet
        # q.epsilon = max(0.01, max_exp_rate - 0.01*(n_epi/200)) # if annealing

        # # decay in reset rates
        # if reset_decay == 'linear':
        #     reset_rate = 0.0 # edit!! add the correct scheme!!
        #     q.epsilon = 0.0
        # elif reset_decay == 'twomodes':
        #     if training_done == True:
        #         reset_rate = 0.0
        #         q.epsilon = 0.0 # also turn off exploration

        s, _ = env.reset(options={'start_loc':start_s, 'goal_loc':goal_s, 'reset_rate':reset_rate})
        done = False # the environment automatically sets done = True when the goal state is reached

        # training (not testing) episode
        while not done:
            a = q.chooseAction(s)
            s_prime, r, done, truncated, info = env.step(a) # resetting is encoded into this
            reset_last_step = info['reset_last_step']

            # if reset last step, and if resetting memory, then wipe memory
            if reset_last_step == True:
                length_this_episode = 0 # reset back to 0
                if resetting_mode == 'memory':
                    q = QLearn(actions, epsilon, learning_rate, gamma) # wipe memory and initialize again

            elif reset_last_step == False: # everything goes normally
                q.learn(s, a, r, s_prime)
                length_this_episode += 1

            s = s_prime
            epilength_this_episode += 1
            # reward_this_episode += r * gamma**(epilength_this_episode) # DISCOUNTED reward is a more accurate metric
            reward_this_episode -= 1 # simple reward fn. decreases with #steps

            if done:
                total_reward_vec = np.append(total_reward_vec, reward_this_episode)
                total_epilength_vec = np.append(total_epilength_vec, epilength_this_episode)
                total_length_vec = np.append(total_length_vec, length_this_episode)
                regret_this_episode = rwd_opt - reward_this_episode
                total_regret_vec = np.append(total_regret_vec, regret_this_episode)
                break

        n_epi += 1

        # do another run with exploration and resetting set to 0 to test how good the noiseless solution is
        if learning_end_condition == 'threshold':
            q.epsilon = 0.0 # also turn off exploration
            # turn off resetting for the test runs
            s, _ = env.reset(options={'start_loc':start_s, 'goal_loc':goal_s, 'reset_rate':0.0})
            done = False # the environment automatically sets done = True when the goal state is reached

            # initialize values to be potentially appended
            reward_test_episode = 0
            epilength_test_episode = 0

            QTable_direction_incorrect = 0 # increment upwards
            max_num_steps = int(taxicab_length * 1.1) # this needs to be exactly the taxicab length -- otherwise possible for training to never be done
            # testing_step_no = 0 # incremented up

            while epilength_test_episode <= max_num_steps:
                # run another episode
                a = q.chooseAction(s)
                s_prime, r, done, truncated, info = env.step(a) # resetting is encoded into this
                epilength_test_episode += 1
                reward_test_episode -= 1

                if q.QDirectional(s) == False:
                    QTable_direction_incorrect += 1

                # DEBUGGING
                # QTable_direction_correct = QTable_direction_correct and q.QDirectional(s) # only true if everything is true # old, and doesn't allow for error
                # print(testing_step_no) # debug
                # print("Q-table count:", q.getCount(s)) # debug
                # print(q.QDirectional(s)) # debug

                # if q.QDirectional(s) == False: # debug
                #     q.print_Q_table(s)

                s = s_prime
                if done: 
                    break

            # if done within 1.1 * taxicab, then learning has "completed" and we can turn it off
            if done and QTable_direction_incorrect <= (max_num_steps - taxicab_length): # roughly the same tolerance for the QTable and step lengths
                training_done = True
                training_done_epi = n_epi
                length_test_episode = epilength_test_episode # no resetting
                regret_test_episode = rwd_opt - reward_test_episode

                total_reward_vec = np.append(total_reward_vec, reward_test_episode)
                total_epilength_vec = np.append(total_epilength_vec, epilength_test_episode)
                total_length_vec = np.append(total_length_vec, length_test_episode)
                total_regret_vec = np.append(total_regret_vec, regret_test_episode)
                print(f"Training completed at episode {training_done_epi}")

        # can also check whether training is done using the condition of subsequent stability of the Q-table
        elif learning_end_condition == 'QStable':
            # check if Q-values are stable after this given episode
            stable_now = q.QStable()
            stable_window.append(stable_now)

            if len(stable_window) == n_stable and all(stable_window):
                training_done = True
                training_done_epi = n_epi - n_stable
                # print(f"Training completed at episode {training_done_epi}") # debug

            # save the max q indices to compare Q tables
            q.save_max_q_indices()
            # DEBUG: print the Q-table at the end of QStable to check if the maximum is unique

    # save stored vectors to feed into bash script, which then writes them to one CSV file
    np.save(total_reward_vec_file, total_reward_vec)
    np.save(total_epilength_vec_file, total_epilength_vec)
    np.save(total_length_vec_file, total_length_vec)
    np.save(total_regret_vec_file, total_regret_vec)
    np.save(total_training_done_epi_file, training_done_epi) # luckily, this (+1?) IS the number of episodes
    # save the ending regret -- this needs to be different than the non-adjustable case!
    np.save(ending_regret_file, regret_test_episode) # ending regret is the last regret_this_episode
    env.close()

if __name__ == '__main__':
    main()