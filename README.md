Summary of files: 

- learning_with_resets.py: main RL code, which draws from the class QLearn in learning_algorithms/QLearn.py
- learning_with_resets.py creates an environment according to the code snippet in the image below, and draws from gymnasium environments from simplegrid_with_resets/envs/

![image](https://github.com/user-attachments/assets/753d74a3-c488-40bb-9c9d-96ef704cd3b5)
- to enable the movie, change the "render_mode=None" above to "render_mode="human""
- to run learning_with_resets.py: "python learning_with_resets.py --reset_rate ResetRate --num_episodes NumEpisodes"
- sweep_learning_resets.sh currently performs a parameter sweep over 6 reset rates between 0 and 0.05, with 10 trials for each unique rate, and saves arrays for the reward per episode and length of episode
- plotting.py attempts to plot episode number versus average (reward per episode/length of episode), but still needs debugging
