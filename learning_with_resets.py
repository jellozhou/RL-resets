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
    parser.add_argument('--qlearn_after_resets',type=str, default='True')
    parser.add_argument('--reset_decay', type=str, default='none')
    parser.add_argument('--n_stable', type=int, default=20)
    parser.add_argument('--resetting_mode', type=str, required=True) # resetting mode -- position / memory
    parser.add_argument('--N', type=int, required=True) # system size
    parser.add_argument('--boundary', type=str, default='fixed')

    args = parser.parse_args()
    reset_rate = args.reset_rate
    learning_rate = args.learning_rate
    gamma = args.gamma
    epsilon = args.epsilon
    num_episodes = args.num_episodes
    # parse render mode argument
    if args.render_mode == "None":
        render_mode_arg = None
        # render_mode_arg = 'human' # to visualize
    else:
        render_mode_arg = args.render_mode # all other options have str inputs
    
    if args.qlearn_after_resets == "True":
        qlearn_after_resets = True
    elif args.qlearn_after_resets == "False":
        qlearn_after_resets = False

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
    
    total_reward_vec = np.empty(num_episodes)
    total_epilength_vec = np.empty(num_episodes)
    total_length_vec = np.empty(num_episodes)
    total_regret_vec = np.empty(num_episodes)

    # initialize reward, regret, epilength vector filenames before we modify the reset_rate, epsilon, etc
    total_reward_vec_file = f"vectors/total_reward_vec_resetrate_{reset_rate}_size_{N}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_qlearnreset_{qlearn_after_resets}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
    total_epilength_vec_file = f"vectors/total_epilength_vec_resetrate_{reset_rate}_size_{N}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_qlearnreset_{qlearn_after_resets}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
    
    # distinguish between episode length and LENGTH, which resets every reset (i.e. it is the eventual path from the start to the goal that the agent finds)
    total_length_vec_file = f"vectors/total_length_vec_resetrate_{reset_rate}_size_{N}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_qlearnreset_{qlearn_after_resets}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
    total_regret_vec_file = f"vectors/total_regret_vec_resetrate_{reset_rate}_size_{N}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_qlearnreset_{qlearn_after_resets}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
    total_training_done_epi_file = f"vectors/training_done_epi_resetrate_{reset_rate}_size_{N}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_qlearnreset_{qlearn_after_resets}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"
    ending_regret_file = f"vectors/ending_regret_file_resetrate_{reset_rate}_size_{N}_boundary_{boundary_type}_learningrate_{learning_rate}_gamma_{gamma}_epsilon_{epsilon}_nstable_{n_stable}_qlearnreset_{qlearn_after_resets}_resetdecay_{reset_decay}_resettingmode_{resetting_mode}.npy"

    env = gym.make(
        system_geom, 
        obstacle_map=create_array(N, obstacle_prob, start_x, start_y, goal_x, goal_y), 
        render_mode=render_mode_arg
    )
    
    actions = list(env.unwrapped.MOVES.keys())
    q = QLearn(actions, epsilon, learning_rate, gamma) # initialize
    training_done = False # whether, at the given episode, training has completed
    stable_window = deque(maxlen=n_stable)

    training_done_epi = -1
    for n_epi in range(num_episodes):
        print("Current episode number: ", n_epi)
        # initialize reward, episode length, regret
        reward_this_episode = 0
        epilength_this_episode = 0
        length_this_episode = 0
        # is there any value in calculating the step-wise integrated regret?
        regret_this_episode = 0

        # set epsilon values according to the decay option
        q.epsilon = epsilon # no annealing yet
        # q.epsilon = max(0.01, max_exp_rate - 0.01*(n_epi/200)) # if annealing

        # decay in reset rates
        if reset_decay == 'linear':
            reset_rate = 0.0 # edit!! add the correct scheme!!
            q.epsilon = 0.0
        elif reset_decay == 'twomodes':
            if training_done == True:
                reset_rate = 0.0
                q.epsilon = 0.0 # also turn off exploration

        s, _ = env.reset(options={'start_loc':start_s, 'goal_loc':goal_s, 'reset_rate':reset_rate})
        # optimal reward determined by start loc and goal loc alone
        # rwd_opt = gamma**(np.absolute(goal_x - start_x) + np.absolute(goal_y - start_y)) # reward 1 at the end state -- multiply by discount factor
        # change optimal reward to be literally the shortest path in taxicab
        rwd_opt = -1*(np.absolute(goal_x - start_x) + np.absolute(goal_y - start_y))
        # col = s % N
        # row = s // N
        done = False

        while not done:
            a = q.chooseAction(s)
            s_prime, r, done, truncated, info = env.step(a) # resetting is encoded into this
            reset_last_step = info['reset_last_step']

            # if reset last step, and if resetting memory, then wipe memory
            if reset_last_step == True:
                length_this_episode = 0 # reset back to 0
                if resetting_mode == 'memory':
                    q = QLearn(actions, epsilon, learning_rate, gamma) # wipe memory and initialize again
                # whether or not to qlearn
                # only skip learning step -- the reward, length, and regret calculations can remain
                if qlearn_after_resets == True:
                    q.learn(s, a, r, s_prime)
            # print(reset_last_step)

            elif reset_last_step == False: # everything goes normally
                q.learn(s, a, r, s_prime)
                length_this_episode += 1

            s = s_prime
            epilength_this_episode += 1
            # reward_this_episode += r * gamma**(epilength_this_episode) # DISCOUNTED reward is a more accurate metric
            reward_this_episode -= 1 # simple reward fn. decreases with #steps

            if done:
                total_reward_vec[n_epi] = reward_this_episode
                total_epilength_vec[n_epi] = epilength_this_episode
                total_length_vec[n_epi] = length_this_episode
                regret_this_episode = rwd_opt - reward_this_episode
                total_regret_vec[n_epi] = regret_this_episode
                break

        # check if Q-values are stable after this given episode
        stable_now = q.QStable()
        stable_window.append(stable_now)

        if len(stable_window) == n_stable and all(stable_window):
            if not training_done:
                training_done = True
                training_done_epi = n_epi - n_stable
                print(f"Training completed at episode {training_done_epi}")

        # if q.QStable():
        #     # print(training_done)
        #     if training_done == False: # first such time
        #         training_done_epi = n_epi
        #         training_done = True
        #         print(training_done_epi)

        # save the max q indices to compare Q tables
        q.save_max_q_indices()

    # save stored vectors to feed into bash script, which then writes them to one CSV file
    np.save(total_reward_vec_file, total_reward_vec)
    np.save(total_epilength_vec_file, total_epilength_vec)
    np.save(total_length_vec_file, total_length_vec)
    np.save(total_regret_vec_file, total_regret_vec)
    np.save(total_training_done_epi_file, training_done_epi)
    # save the ending regret
    np.save(ending_regret_file, regret_this_episode) # ending regret is the last regret_this_episode

    env.close()

if __name__ == '__main__':
    main()