import numpy as np
import torch


class episode:

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.collisions = []
        self.next_states = []
        self.goals = []

    def append(self, state, action, reward, done, collision, next_state, goal):
        # self.states.append(state)
        # self.actions.append(action)
        # self.rewards.append(reward)
        # self.dones.append(done)
        # self.collisions.append(collision)
        # self.next_states.append(next_state)
        # self.goals.append(goal)

        self.states.append(torch.tensor(state))
        self.actions.append(torch.tensor(action))
        self.rewards.append(torch.tensor(reward))
        self.dones.append(torch.tensor(done))
        self.collisions.append(torch.tensor(collision))
        self.next_states.append(torch.tensor(next_state))
        self.goals.append(torch.tensor(goal))


    def size(self):
        return len(self.states)

    def __call__(self):
        return [
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.rewards),
            torch.stack(self.dones),
            torch.stack(self.collisions),
            torch.stack(self.next_states),
            torch.stack(self.goals)
            ]
        # return np.array([
        #     self.states,
        #     self.actions,
        #     self.rewards,
        #     self.dones,
        #     self.collisions,
        #     self.next_states,
        #     self.goals
        #     ], dtype=object)

    def resample_goals(self, env, num_subgoals=4):
        idxes = np.random.randint(low=0, high=self.size(), size=num_subgoals)

        new_episodes = [self.recreate_episode(env, id) for id in idxes]

        return new_episodes

    def recreate_episode(self, env, id_subgoal):
        new_episode = episode()
        new_episode.states = self.states[:id_subgoal][:]
        new_episode.actions = self.actions[:id_subgoal][:]
        new_episode.dones = self.dones[:id_subgoal][:]
        new_episode.collisions = self.collisions[:id_subgoal][:]
        new_episode.next_states = self.next_states[:id_subgoal][:]
        
        goal = self.next_states[id_subgoal][:]
        new_episode.goals = [goal] * id_subgoal

        # new_episode.rewards = [env.calc_reward(c, s, goal) for c, s in zip(np.array(new_episode.collisions), np.array(new_episode.states))]
        new_episode.rewards = [torch.tensor(env.calc_reward(c, s, goal)) for c, s in zip(new_episode.collisions, new_episode.states)]

        return new_episode

class EpisodeMemory:

    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.size = 0
        self.episodes = []

    def append(self, episode, num_subgoals=4):
        self.size += episode.size()
        self.episodes.append(episode())

        for ep in episode.resample_goals(env, num_subgoals):
            self.episodes.append(ep())

        while self.size > self.mem_size:
            self.size -= self.episodes[0].size()
            del self.episodes[0]

    def extend(self, memory):
        self.size += memory.size
        self.episodes.extend(memory.episodes)

        while self.size > self.mem_size:
            self.size -= self.episodes[0].size()
            del self.episodes[0]

    def flatten(self):
        # return np.concatenate(self.episodes, -1)
        return torch.cat(self.episodes, -1)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self.size, size=batch_size)
        flatten_episodes = self.flatten()
        # return [
        #     torch.stack([torch.tensor(data, dtype=torch.float) for data in flatten_episodes[0, idxes]]),
        #     torch.stack([torch.tensor(data, dtype=torch.float) for data in flatten_episodes[1, idxes]]),
        #     torch.stack([torch.tensor(data, dtype=torch.float) for data in flatten_episodes[2, idxes]]),
        #     torch.stack([torch.tensor(data, dtype=torch.float) for data in flatten_episodes[3, idxes]]),
        #     torch.stack([torch.tensor(data, dtype=torch.float) for data in flatten_episodes[5, idxes]]),
        #     torch.stack([torch.tensor(data, dtype=torch.float) for data in flatten_episodes[6, idxes]]),
        #     ]
        return [flatten_episodes[i, idxes] for i in [0, 1, 2, 3, 5, 6]]

if __name__ == '__main__':

    import numpy as np
    import pprint
    import random
    from maze3Env import maze3Env 
    env = maze3Env()
    env.setting()

    episode0 = episode()

    episode0.append(np.random.rand(2), np.random.rand(3), random.random(), False, False, np.random.rand(2), np.random.rand(2))
    episode0.append(np.random.rand(2), np.random.rand(3), random.random(), False, False, np.random.rand(2), np.random.rand(2))
    episode0.append(np.random.rand(2), np.random.rand(3), random.random(), False, False, np.random.rand(2), np.random.rand(2))
    episode0.append(np.random.rand(2), np.random.rand(3), random.random(), False, False, np.random.rand(2), np.random.rand(2))
    episode0.append(np.random.rand(2), np.random.rand(3), random.random(), False, False, np.random.rand(2), np.random.rand(2))
    episode0.append(np.random.rand(2), np.random.rand(3), random.random(), False, False, np.random.rand(2), np.random.rand(2))
    episode0.append(np.random.rand(2), np.random.rand(3), random.random(),  True, False, np.random.rand(2), np.random.rand(2))

    episode1 = episode()

    episode1.append(np.random.rand(2), np.random.rand(3), random.random(), False, False, np.random.rand(2), np.random.rand(2))
    episode1.append(np.random.rand(2), np.random.rand(3), random.random(), False, False, np.random.rand(2), np.random.rand(2))
    episode1.append(np.random.rand(2), np.random.rand(3), random.random(),  True, False, np.random.rand(2), np.random.rand(2))

    memory = EpisodeMemory(mem_size=1000)
    memory.append(episode0)
    memory.append(episode1)

    # print(memory.episodes)

    # print()
    # print(memory.flatten())
    # print(memory.flatten().shape)
    # print(memory.flatten()[0].shape)
    
    print("------------------------------")

    print()
    pprint.pprint(memory.sample(3))