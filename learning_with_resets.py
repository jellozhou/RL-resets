import collections
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

# import gym_simplegrid
# from simplegrid_with_resets.envs import SimpleGridEnv, SimpleGridEnvResets

#Hyperparameters
learning_rate = 0.0005 # alpha
gamma         = 0.98 # closer to 1?
N = 20 # box side length
# reset_rate = 0.01 # parameter sweep
obstacle_prob = 0. # prob that cell in box initialized as obstacle
max_exp_rate = 0.08 # exploration vs exploitation rate, not sure if this or the linear annealing is better
# num_episodes = 100 # specified in parameter sweep

# convert starting & goal (x,y) to state
start_x = int(N//3)
start_y = int(N//3)
goal_x = int(2*N//3)
goal_y = int(2*N//3)
start_s = start_x + start_y*N
goal_s = goal_x + goal_y*N

def create_array(N, obstacle_prob):
    # Initialize an N x N array filled with zeros
    arr = np.random.choice([0, 1], size=(N, N), p=[1 - obstacle_prob, obstacle_prob])

    # iterate until there is a path from start to goal
    while find_path(arr, (start_x, start_y), (goal_x, goal_y)) == False:
        arr = np.random.choice([0, 1], size=(N, N), p=[1 - obstacle_prob, obstacle_prob])

    # Convert the array to the string form
    return [''.join(map(str, row)) for row in arr]

def main():

    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset_rate', type=float, required=True)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--render_mode', type=str, default=None)
    # there is some issue with stdin bool arguments
    parser.add_argument('--qlearn_after_resets',type=str, default='True')

    args = parser.parse_args()
    reset_rate = args.reset_rate
    num_episodes = args.num_episodes
    # parse render mode argument
    if args.render_mode == "None":
        render_mode_arg = None
    else:
        render_mode_arg = args.render_mode # all other options have str inputs
    
    if args.qlearn_after_resets == "True":
        qlearn_after_resets = True
    elif args.qlearn_after_resets == "False":
        qlearn_after_resets = False
    
    total_reward_vec = np.empty(num_episodes)
    total_epilength_vec = np.empty(num_episodes)
    total_regret_vec = np.empty(num_episodes)

    env = gym.make(
        'SimpleGridReset-v0', 
        obstacle_map=create_array(N, obstacle_prob), 
        render_mode=render_mode_arg
    )
    
    actions = list(env.MOVES.keys())
    epsilon = max_exp_rate
    q = QLearn(actions, epsilon, learning_rate, gamma)

    for n_epi in range(num_episodes):
        print("Current episode number: ", n_epi)
        # initialize reward, episode length, regret
        reward_this_episode = 0
        epilength_this_episode = 0
        # is there any value in calculating the step-wise integrated regret?
        regret_this_episode = 0

        # two ways to set up exploration vs exploitation strategy
        # (1) linear annealing; update epsilon at every step
        # max_exp_rate and '/200' can both be tuned
        q.epsilon = max(0.01, max_exp_rate - 0.01*(n_epi/200))
        # (2) constant rate of exploration
        # epsilon = max_exp_rate

        s, _ = env.reset(options={'start_loc':start_s, 'goal_loc':goal_s, 'reset_rate':reset_rate})
        # col = s % N
        # row = s // N
        done = False

        while not done:
            a = q.chooseAction(s)
            s_prime, r, done, truncated, info = env.step(a)
            reset_last_step = info['reset_last_step']
            # print(reset_last_step)

            # inner loop to not learn after resetting, only if the flag is there
            # only skip learning step -- the reward, length, and regret calculations can remain
            if reset_last_step == True:
                if qlearn_after_resets == True:
                    q.learn(s, a, r, s_prime)
            elif reset_last_step == False:
                q.learn(s, a, r, s_prime)
            s = s_prime
            reward_this_episode += r
            epilength_this_episode += 1
            # optimal reward per episode: calculate manually
            optimal_reward_this_step = 0
            if env.almost_goal():
                optimal_reward_this_step = 1
            regret_this_episode += (optimal_reward_this_step - r)

            if done:
                total_reward_vec[n_epi] = reward_this_episode
                total_epilength_vec[n_epi] = epilength_this_episode
                total_regret_vec[n_epi] = regret_this_episode
                break

    # save stored vectors to feed into bash script, which then writes them to one CSV file
    np.save("total_reward_vec.npy", total_reward_vec)
    np.save("total_epilength_vec.npy", total_epilength_vec)
    np.save("total_regret_vec.npy", total_regret_vec)
    env.close()

if __name__ == '__main__':
    main()