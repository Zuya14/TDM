import torch
import numpy as np


# class episode:
#     def __init__(self, device):
#         self.states = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []
#         self.collisions = []
#         self.next_states = []
#         self.goals = []

#         self.device=device

#     def append(self, state, action, reward, done, collision, next_state, goal):
#         self.states.append(state)
#         self.actions.append(action)
#         self.rewards.append(reward)
#         self.dones.append(done)
#         self.collisions.append(collision)
#         self.next_states.append(next_state)
#         self.goals.append(goal)

#         # self.states.append(torch.tensor(state))
#         # self.actions.append(torch.tensor(action))
#         # self.rewards.append(torch.tensor(reward))
#         # self.dones.append(torch.tensor(done))
#         # self.collisions.append(torch.tensor(collision))
#         # self.next_states.append(torch.tensor(next_state))
#         # self.goals.append(torch.tensor(goal))

#     def size(self):
#         return len(self.states)

#     def __call__(self):
#         return np.array([
#         self.states,
#         self.actions,
#         self.rewards,
#         self.dones,
#         self.collisions,
#         self.next_states,
#         self.goals
#         ], dtype=object)

#     def resample_goals(self, env, num_subgoals=4):
#         idxes = np.random.randint(low=0, high=self.size(), size=num_subgoals)

#         new_episodes = [self.recreate_episode(env, id) for id in idxes]

#         return new_episodes

#     def recreate_episode(self, env, id_subgoal):
#         new_episode = episode(self.device)
#         new_episode.states = self.states[:id_subgoal]
#         new_episode.actions = self.actions[:id_subgoal]
#         new_episode.dones = self.dones[:id_subgoal]
#         new_episode.collisions = self.collisions[:id_subgoal]
#         new_episode.next_states = self.next_states[:id_subgoal]
        
#         goal = self.next_states[id_subgoal]
#         new_episode.goals = [goal] * id_subgoal

#         new_episode.rewards = [env.calc_reward(c, s, goal) for c, s in zip(new_episode.collisions, new_episode.states)]

#         return new_episode

class episode:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.goals = []
        self.collisions = []

    def append(self, state, action, reward, done, next_state, goal, collision):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.goals.append(goal)
        self.collisions.append(collision)

    def size(self):
        return len(self.states)

    def __call__(self):
        return [
        self.states,
        self.actions,
        self.rewards,
        self.dones,
        self.next_states,
        self.goals,
        self.collisions
        ]

class HindsightReplayBuffer():

    def __init__(self, buffer_size, state_size, action_size, goal_size, device, num_subgoals=4):
        self.device=device
        self.episode = episode()
        self.num_subgoals = num_subgoals

        # 次にデータを挿入するインデックス．
        self._p = 0
        # データ数．
        self._n = 0
        # リプレイバッファのサイズ．
        self.buffer_size = buffer_size

        # GPU上に保存するデータ．
        self.states = torch.empty((buffer_size, *state_size), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_size), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_size), dtype=torch.float, device=device)
        self.goals = torch.empty((buffer_size, *goal_size), dtype=torch.float, device=device)
        self.collisions = torch.empty((buffer_size, 1), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state, goal, collision, save_episode=True):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.goals[self._p].copy_(torch.from_numpy(goal))
        self.collisions[self._p] = float(collision)

        if save_episode:
            self.episode.append(state, action, reward, done, next_state, goal, collision)

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
            self.next_states[indices],
            self.goals[indices]
        )

    def resample_goals(self, env):
        states, actions, rewards, dones, next_states, goals, collisions = self.episode()
        episode_len = self.episode.size()
        
        for i in range(episode_len-1):
            indices = np.random.randint(low=i, high=episode_len-1, size=self.num_subgoals)

            for index in indices:
                new_goal = next_states[index]
                if not collisions[i]:
                    self.append(states[i], actions[i], env.calc_reward(collisions[i], states[i], new_goal), dones[i] or env.sim.isArrive(new_goal, next_states[i]), next_states[i], new_goal, collisions[i], False)

        self.episode = episode()




# class HindsightReplayBuffer():

#     def __init__(self, buffer_size, device):
#         self.buffer_size = buffer_size
#         self.size = 0
#         self.device=device

#         self.episodes = []

#     def append(self, episode, env, num_subgoals=4):
#         self.size += episode.size()
#         self.episodes.append(episode())

#         for ep in episode.resample_goals(env, num_subgoals):
#             self.episodes.append(ep())

#         while self.size > self.buffer_size:
#             self.size -= self.episodes[0].size()
#             del self.episodes[0]

#     def extend(self, memory):
#         self.size += memory.size
#         self.episodes.extend(memory.episodes)

#         while self.size > self.buffer_size:
#             self.size -= self.episodes[0].size()
#             del self.episodes[0]

#     def flatten(self):
#         return np.concatenate(self.episodes, -1)

#     def sample(self, batch_size):
#         idxes = np.random.randint(low=0, high=self.size, size=batch_size)
#         flatten_episodes = self.flatten()
#         return [
#             torch.stack([torch.tensor(data, dtype=torch.float, device=self.device) for data in flatten_episodes[0, idxes]]),
#             torch.stack([torch.tensor(data, dtype=torch.float, device=self.device) for data in flatten_episodes[1, idxes]]),
#             torch.stack([torch.tensor(data, dtype=torch.float, device=self.device) for data in flatten_episodes[2, idxes]]),
#             torch.stack([torch.tensor(data, dtype=torch.float , device=self.device) for data in flatten_episodes[3, idxes]]),
#             torch.stack([torch.tensor(data, dtype=torch.float, device=self.device) for data in flatten_episodes[5, idxes]]),
#             torch.stack([torch.tensor(data, dtype=torch.float, device=self.device) for data in flatten_episodes[6, idxes]]),
#             ]

class HindsightReplayBuffer_old():

    def __init__(self, buffer_size, state_size, action_size, goal_size, device):
        # 次にデータを挿入するインデックス．
        self._p = 0
        # データ数．
        self._n = 0
        # リプレイバッファのサイズ．
        self.buffer_size = buffer_size

        # GPU上に保存するデータ．
        self.states = torch.empty((buffer_size, *state_size), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_size), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_size), dtype=torch.float, device=device)
        self.goals = torch.empty((buffer_size, *goal_size), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state, goal):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.goals[self._p].copy_(torch.from_numpy(goal))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes],
            self.goals[idxes]
        )

class TDM_HindsightReplayBuffer():

    def __init__(self, buffer_size, state_size, action_size, goal_size, device):
        # 次にデータを挿入するインデックス．
        self._p = 0
        # データ数．
        self._n = 0
        # リプレイバッファのサイズ．
        self.buffer_size = buffer_size

        # GPU上に保存するデータ．
        self.states = torch.empty((buffer_size, *state_size), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_size), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_size), dtype=torch.float, device=device)
        self.goals = torch.empty((buffer_size, *goal_size), dtype=torch.float, device=device)
        self.num_steps_lefts = torch.empty((buffer_size, 1), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state, goal, num_steps_left):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.goals[self._p].copy_(torch.from_numpy(goal))
        self.num_steps_lefts[self._p].copy_(torch.from_numpy(num_steps_left))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes],
            self.goals[idxes],
            self.num_steps_lefts[idxes]
        )