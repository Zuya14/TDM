from DDPG import DDPG, ActorNetwork#, CriticNetwork
from HindsightReplayBuffer import TDM_ReplayBuffer, episode
import torch
import numpy as np
import random
import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, goal_size, hidden_size=64):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, 1),
            # nn.Linear(hidden_size, goal_size),
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, 1),
            # nn.Linear(hidden_size, goal_size),
        )

    def forward(self, states, actions, goals):
        x = torch.cat([states, actions], dim=-1)
        return self.net1(x), self.net2(x)
        # return -torch.abs(goals - self.net1(x)),  -torch.abs(goals - self.net2(x))
        # return -torch.abs(goals - self.net1(x)).sum(),  -torch.abs(goals - self.net2(x)).sum()

class TDM(DDPG):

    def __init__(self, state_size, action_size, goal_size, hidden_size=64, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 batch_size=256, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3,
                 replay_size=10**6, start_steps=10**4, tau=5e-3, alpha=0.2, reward_scale=1.0, epsilon_decay = 50000):

        super().__init__(
            state_size,
            action_size, 
            hidden_size, 
            device,
            batch_size, 
            gamma, 
            lr_actor, 
            lr_critic,
            replay_size, 
            start_steps, 
            tau, 
            alpha, 
            reward_scale,
            epsilon_decay
        )

        self.name = 'TDM'

        self.buffer = TDM_ReplayBuffer(
            buffer_size=replay_size,
            state_size=state_size,
            action_size=action_size,
            goal_size=goal_size,
            device=device
        )

        # Actor-Criticのネットワークを構築する．
        self.actor = ActorNetwork(
            state_size=state_size[0]+goal_size[0]+1,
            action_size=action_size[0],
            hidden_size=hidden_size
        ).to(device)
        self.critic = CriticNetwork(
            state_size=state_size[0]+goal_size[0]+1,
            action_size=action_size[0],
            goal_size=goal_size[0],
            hidden_size=hidden_size
        ).to(device)
        self.critic_target = CriticNetwork(
            state_size=state_size[0]+goal_size[0]+1,
            action_size=action_size[0],
            goal_size=goal_size[0],
            hidden_size=hidden_size
        ).to(device).eval()

        # ターゲットネットワークの重みを初期化し，勾配計算を無効にする．
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # オプティマイザ．
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.device = device
    
    def exploit(self, state, goal, num_steps_left):
        """ 決定論的な行動を返す． """
        # state = torch.tensor(np.concatenate([state, goal]), dtype=torch.float, device=self.device).unsqueeze_(0)
        # state = torch.tensor(np.concatenate([state, goal-state, np.array([num_steps_left])]), dtype=torch.float, device=self.device).unsqueeze_(0)
        state = torch.tensor(np.concatenate([state, goal-state, np.array([np.log(num_steps_left+2)])]), dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    def step(self, env, state, goal, t, steps):
        t += 1

        num_steps_left = env.get_left_steps()

        # 学習初期の一定期間(start_steps)は，ランダムに行動して多様なデータの収集を促進する．
        if steps <= self.start_steps:
            action = env.action_space.sample()
        else:
            if random.random() < self.epsilon_func(steps):
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = self.exploit(state, goal, num_steps_left)

        next_state, reward, done, _ = env.step(action)

        # ゲームオーバーではなく，最大ステップ数に到達したことでエピソードが終了した場合は，
        # 本来であればその先も試行が継続するはず．よって，終了シグナルをFalseにする．
        # NOTE: ゲームオーバーによってエピソード終了した場合には， done_masked=True が適切．
        # しかし，以下の実装では，"たまたま"最大ステップ数でゲームオーバーとなった場合には，
        # done_masked=False になってしまう．
        # その場合は稀で，多くの実装ではその誤差を無視しているので，今回も無視する．
        if t == env._max_episode_steps:
            done_masked = False
        else:
            done_masked = done

        # リプレイバッファにデータを追加する．
        self.buffer.append(state, action, reward, done_masked, next_state, goal, env.sim.isContacts(), num_steps_left)

        # エピソードが終了した場合には，環境をリセットする．
        if done:
            t = 0
            next_state = env.reset()

            self.buffer.resample_goals(env)

        return next_state, t

    def update(self):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states, goals, num_steps_left = self.buffer.sample(self.batch_size)
        # states, actions, rewards, dones, next_states, goals = self.buffer.sample(self.batch_size)
        # states, actions, rewards, dones, collisions, next_states, goals = self.buffer.sample(self.batch_size)

        self.update_critic(states, actions, rewards, dones, next_states, goals, num_steps_left)
        self.update_actor(states, goals, num_steps_left)
        self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states, goals, num_steps_left):
        # states2 = torch.cat([states, goals], dim=-1)
        # states2 = torch.cat([states, goals-states, num_steps_left], dim=-1)
        states2 = torch.cat([states, goals-states, torch.log(num_steps_left+2)], dim=-1)
        curr_qs1, curr_qs2 = self.critic(states2, actions, goals)

        with torch.no_grad():
            # next_states2 = torch.cat([next_states, goals], dim=-1)
            # next_states2 = torch.cat([next_states, goals-next_states], dim=-1)
            # next_states2 = torch.cat([next_states, goals-next_states, num_steps_left-1], dim=-1)
            next_states2 = torch.cat([next_states, goals-next_states,  torch.log(num_steps_left+2-1)], dim=-1)
            next_actions = self.actor(next_states2)

            next_qs1, next_qs2 = self.critic_target(next_states2, next_actions, goals)
            next_qs = torch.min(next_qs1, next_qs2)

        terminals = 1 - (1 - dones) * (num_steps_left != 0)
        rewards = rewards * terminals

        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

    def update_actor(self, states, goals, num_steps_left):
        # states2 = torch.cat([states, goals], dim=-1)
        # states2 = torch.cat([states, goals-states, num_steps_left], dim=-1)
        states2 = torch.cat([states, goals-states, torch.log(num_steps_left+2)], dim=-1)
        # actions = self.actor(states2)
        actions, pre_tanh_value = self.actor(states2, return_preactivations=True)
        qs1, qs2 = self.critic(states2, actions, goals)
        
        pre_activation_policy_loss = (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        
        # loss_actor = -torch.min(qs1, qs2).mean()
        loss_actor = -torch.min(qs1, qs2).mean() + pre_activation_policy_loss * 0.01


        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

    def save(self, path="./"):
        torch.save(self.actor.to('cpu').state_dict(), path+"TDM_actor.pth")
        self.actor.to(self.device)

        torch.save(self.critic.to('cpu').state_dict(), path+"TDM_critic.pth")
        self.critic.to(self.device)

    def load(self, path="./"):
        self.actor.load_state_dict(torch.load(path+"TDM_actor.pth"))
        self.critic.load_state_dict(torch.load(path+"TDM_critic.pth"))

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

state_t と　state_t+1 のそれぞれに対して goal を concat するのが下位層だと面倒なので上位層（Bufferくらい）でconcatしたい

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''