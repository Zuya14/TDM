import gym
import pybullet_envs
# from PPO import PPO
from GC_DDPG_HER import GC_DDPG_HER
from trainer import Trainer

# from mazeEnv import mazeEnv 
# from crossEnv import crossEnv 
from square3Env import square3Env 
from maze3Env import maze3Env 

# ENV_ID = 'InvertedPendulumBulletEnv-v0'
SEED = 0
# NUM_STEPS = 5 * 10 ** 4
# NUM_STEPS = 3 * 10 ** 4
# NUM_STEPS = 5 * 10 ** 4
NUM_STEPS = 10 * 10 ** 4
# NUM_STEPS = 2 * 10 ** 5
# NUM_STEPS = 25 * 10 ** 4
# NUM_STEPS = 1 * 10 ** 5
# NUM_STEPS = 3 * 10 ** 5
EVAL_INTERVAL = 10 ** 3

# STARTSTEPS = 1000
STARTSTEPS = 3000

# env = gym.make(ENV_ID)
# env_test = gym.make(ENV_ID)

# env = mazeEnv()
# env = crossEnv()
# env = square3Env()
env = maze3Env()
env.setting()

# env_test = mazeEnv()
# env_test = crossEnv()
# env_test = square3Env()
env_test = maze3Env()
env_test.setting()

algo = GC_DDPG_HER(
    state_size=env.observation_space.shape,
    action_size=env.action_space.shape,
    goal_size = env.observation_space.shape,
    epsilon_decay = NUM_STEPS,
    start_steps=STARTSTEPS
)

trainer = Trainer(
    env=env,
    env_test=env_test,
    algo=algo,
    num_steps=NUM_STEPS,
    eval_interval=EVAL_INTERVAL,
    is_GC=True
)

trainer.train()

trainer.plot()

algo.save()

# trainer.visualize()
