Summary of files: 

- learning_with_resets.py: main RL code, which draws from the class QLearn in learning_algorithms/QLearn.py
- learning_with_resets.py creates an environment according to the code snippet in the image below, and draws from gymnasium environments from simplegrid_with_resets/envs/
- sweep_learning_resets.py sweeps learning_with_resets.py over various hyperparameters that can be adjusted in the code, and also calls the plotting code (plotting.py)
- sweep_learning_resets.sh theoretically does the same thing as the .py file but it doesn't currently work
