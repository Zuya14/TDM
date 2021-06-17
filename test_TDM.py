import gym
import pybullet_envs
# from PPO import PPO
from TDM import TDM
from trainer import Trainer

# from mazeEnv import mazeEnv 
# from crossEnv import crossEnv 
from square3Env import square3Env 
from maze3Env import maze3Env 
import numpy as np

# ENV_ID = 'InvertedPendulumBulletEnv-v0'
SEED = 0
# NUM_STEPS = 5 * 10 ** 4
# NUM_STEPS = 10 * 10 ** 4
NUM_STEPS = 2 * 10 ** 5
# NUM_STEPS = 25 * 10 ** 4
EVAL_INTERVAL = 10 ** 3

# env = gym.make(ENV_ID)
# env_test = gym.make(ENV_ID)

# env_test = mazeEnv()
# env_test = crossEnv()
env_test = square3Env()
# env_test = maze3Env()
env_test.setting()

algo = TDM(
    state_size=env_test.observation_space.shape,
    action_size=env_test.action_space.shape,
    goal_size = env_test.observation_space.shape
)

trainer = Trainer(
    env=env_test,
    env_test=env_test,
    algo=algo,
    num_steps=NUM_STEPS,
    eval_interval=EVAL_INTERVAL,
    is_GC=True
)

algo.load()

trainer.saveVideo()

subgoals = np.array(
    [
        [1.5, 4.5],
        [1.5, 7.5],
        [4.5, 7.5],
        [7.5, 7.5],
        [7.5, 4.5],
        [7.5, 1.5]
    ]
)

trainer.saveVideo_subgoals(subgoals=subgoals, s="_subgoals")

subgoals = np.array(
    [
        [1.5, 7.5],
        [7.5, 7.5],
        [7.5, 1.5]
    ]
)

trainer.saveVideo_subgoals(subgoals=subgoals, s="_subgoals2")
