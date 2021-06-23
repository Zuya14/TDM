import sys

import gym
import numpy as np
import gym.spaces
from gym.utils import seeding
import torch

import pybullet as p
import cv2
import math

import random
import copy

import bullet_lidar
import sim  

class maze3Env(gym.Env):
    global_id = 0

    def __init__(self):
        super().__init__()
        self.seed(seed=random.randrange(10000))
        self.sim = None

    def setting(self, _id=-1, mode=p.DIRECT, sec=0.1):
        if _id == -1:
            self.sim = sim.sim_maze3(maze3Env.global_id, mode, sec)
            maze3Env.global_id += 1
        else:
            self.sim = sim.sim_maze3(_id, mode, sec)

        # self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # self.lidar = self.createLidar()

        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.lidar.shape)
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.lidar.shape[0]+4,))
        self.observation_space = gym.spaces.Box(low=0.0, high=9.0, shape=(2,))

        self.sec = sec

        self._max_episode_steps = 500
        # self._max_episode_steps = 1000
        # self._max_episode_steps = 250

        self.reset()

    def copy(self, _id=-1):
        new_env = maze3Env()
        
        if _id == -1:
            new_env.sim = self.sim.copy(maze3Env.global_id)
            maze3Env.global_id += 1
        else:
            new_env.sim = self.sim.copy(_id)

        new_env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # new_env.lidar = new_env.createLidar()
        # new_env.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_env.lidar.shape)
        # new_env.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(new_env.lidar.shape[0]+4,))
        self.observation_space = gym.spaces.Box(low=0.0, high=9.0, shape=(2,))

        new_env.sec = self.sec

        return new_env

    def reset(self, tgtpos=[7.5, 1.5]):
        assert self.sim is not None, print("call setting!!") 
        self.sim.reset(sec=self.sec, tgtpos=tgtpos)
        return self.observe()

    def test_reset(self, tgtpos=[7.5, 1.5]):
        assert self.sim is not None, print("call setting!!") 
        self.sim.test_reset(sec=self.sec, tgtpos=tgtpos)
        return self.observe()

    # def createLidar(self):
    #     # resolusion = 12
    #     # resolusion = 36
    #     resolusion = 1
    #     deg_offset = 90.
    #     rad_offset = deg_offset*(math.pi/180.0)
    #     startDeg = -180. + deg_offset
    #     endDeg = 180. + deg_offset

    #     # maxLen = 20.
    #     maxLen = 10.
    #     minLen = 0.
    #     return bullet_lidar.bullet_lidar(startDeg, endDeg, resolusion, maxLen, minLen)

    def step(self, action):

        done = self.sim.step(action)

        # observation = self.sim.observe(self.lidar)
        observation = self.sim.observe()
        # self.min_obs = np.min(observation) * self.lidar.maxLen

        reward = self.get_reward()

        done = done or (self.sim.steps == self._max_episode_steps)

        return observation, reward, done, {}

    def get_left_steps(self):
        return self._max_episode_steps - self.sim.steps

    def observe(self):
        return self.sim.observe()
        # return self.sim.observe(self.lidar)

    # def observe2d(self):
    #     return self.sim.observe2d(self.lidar)

    def get_reward(self):
        return self.calc_reward(self.sim.isContacts(), self.sim.observe(), self.sim.tgt_pos)

    def calc_reward(self, contact, pos, tgt_pos):
        # rewardContact = -10000.0 if contact else 0.0
        # rewardContact = -1000.0 if contact else 0.0
        # rewardContact = -100.0 if contact else 0.0
        # rewardContact = -10.0 if contact else 0.0
        # rewardContact = -50.0 if contact else 0.0
        rewardContact = 0.0
        # rewardDistance = - np.linalg.norm(pos - tgt_pos, ord=2)
        # rewardDistance = 1.0 if np.linalg.norm(pos - tgt_pos, ord=2) < 0.1 else 0.0
        # rewardDistance = 0.0 if (not contact) and (np.linalg.norm(pos - tgt_pos, ord=2) < 0.1) else -1.0
        # rewardDistance = 1.0 if (not contact) and self.sim.isArrive(tgt_pos, pos) else 0.0
        # rewardDistance = 0.0 if (not contact) and self.sim.isArrive(tgt_pos, pos) else -1.0
        # rewardDistance = - np.linalg.norm(tgt_pos - pos, ord=2)
        rewardDistance = - np.linalg.norm(tgt_pos - pos, ord=1)
        # rewardDistance = - np.abs(tgt_pos - pos)
        reward = rewardContact + rewardDistance
        # reward = rewardDistance

        return reward

    def render(self, mode='human', close=False):
        # return self.sim.render(self.lidar)
        return self.sim.render()

    def close(self):
        self.sim.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_random_action(self):
        return self.action_space.sample()

    def getState(self):
        return self.sim.getState()

if __name__ == '__main__':
    
    env = maze3Env()
    env.setting()

    i = 0

    while True:
        i += 1
        
        action = np.array([1.0, 1.0, 0.0])

        _, _, done, _ = env.step(action)

        cv2.imshow("env", env.render())
        if done or cv2.waitKey(100) >= 0:
            print(i)
            break