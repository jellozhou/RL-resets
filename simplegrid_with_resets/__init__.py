from gymnasium.envs.registration import register

register(
    id='SimpleGrid-v0',
    entry_point='gym_simplegrid.envs:SimpleGridEnv',
    max_episode_steps=200
)

register(
    id='SimpleGrid-8x8-v0',
    entry_point='gym_simplegrid.envs:SimpleGridEnv',
    max_episode_steps=200,
    kwargs={'obstacle_map': '8x8'},
)

register(
    id='SimpleGrid-4x4-v0',
    entry_point='gym_simplegrid.envs:SimpleGridEnv',
    max_episode_steps=200,
    kwargs={'obstacle_map': '4x4'},
)


# environments with resets
# regular boundary conditions + negative reward at boundary
register(
    id='SimpleGridReset-v0',
    entry_point='simplegrid_with_resets.envs:SimpleGridEnvResets',
    max_episode_steps=200
)

# periodic boundary conditions
register(
    id='SimpleGridResetPBC-v0',
    entry_point='simplegrid_with_resets.envs:SimpleGridEnvResetsPBC',
    max_episode_steps=200
)

# 1D (line)
register(
    id='SimpleLineReset',
    entry_point='simplegrid_with_resets.envs:SimpleLineEnvResets',
    max_episode_steps=200
)