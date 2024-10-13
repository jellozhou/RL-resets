Summary of files: 

- learning_with_resets.py: main RL code. It draws from the class QLearn in learning_algorithms/QLearn.py.
- learning_with_resets.py creates an environment according to the image below, and draws from gymnasium environments from simplegrid_with_resets/envs/
![image](https://github.com/user-attachments/assets/753d74a3-c488-40bb-9c9d-96ef704cd3b5)
- to enable the movie, change the "render_mode=None" to "render_mode="human"
- to run learning_with_resets.py: "python learning_with_resets.py --reset_rate ResetRate --num_episodes NumEpisodes"
