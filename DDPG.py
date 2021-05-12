import torch
import torch.nn as nn
import torch.nn.functional as F
from ReplayBuffer import ReplayBuffer
import random

# actorのネットワーク
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, states):
        return torch.tanh(self.net(states))

# criticのネットワーク（状態と行動を入力にしてQ値を出力）
class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net1(x), self.net2(x)

class DDPG:

    def __init__(self, state_size, action_size, hidden_size=256, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 batch_size=256, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3,
                 replay_size=10**6, start_steps=10**4, tau=5e-3, alpha=0.2, reward_scale=1.0, epsilon_decay = 50000):

        super().__init__()

        # リプレイバッファ．
        self.buffer = ReplayBuffer(
            buffer_size=replay_size,
            state_size=state_size,
            action_size=action_size,
            device=device,
        )

        # Actor-Criticのネットワークを構築する．
        self.actor = ActorNetwork(
            state_size=state_size[0],
            action_size=action_size[0],
            hidden_size=hidden_size
        ).to(device)
        self.critic = CriticNetwork(
            state_size=state_size[0],
            action_size=action_size[0],
            hidden_size=hidden_size
        ).to(device)
        self.critic_target = CriticNetwork(
            state_size=state_size[0],
            action_size=action_size[0],
            hidden_size=hidden_size
        ).to(device).eval()

        # ターゲットネットワークの重みを初期化し，勾配計算を無効にする．
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # オプティマイザ．
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # その他パラメータ．
        self.action_size = action_size
        self.learning_steps = 0
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.start_steps = start_steps
        self.tau = tau
        self.alpha = alpha
        self.reward_scale = reward_scale

        epsilon_begin = 1.0
        epsilon_end = 0.01
        # epsilon_beginから始めてepsilon_endまでepsilon_decayかけて線形に減らす
        self.epsilon_func = lambda step: max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))

    def is_update(self, steps):
        # 学習初期の一定期間(start_steps)は学習しない．
        return steps >= max(self.start_steps, self.batch_size)

    def exploit(self, state):
        """ 決定論的な行動を返す． """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    def step(self, env, state, t, steps):
        t += 1

        # 学習初期の一定期間(start_steps)は，ランダムに行動して多様なデータの収集を促進する．
        if steps <= self.start_steps:
            action = env.action_space.sample()
        else:
            if random.random() < self.epsilon_func(steps):
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = self.exploit(state)

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
        self.buffer.append(state, action, reward, done_masked, next_state)

        # エピソードが終了した場合には，環境をリセットする．
        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)
        self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2)
        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

    def update_actor(self, states):
        actions = self.actor(states)
        qs1, qs2 = self.critic(states, actions)
        loss_actor = -torch.min(qs1, qs2).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)

    def save(self, path="./"):
        torch.save(self.actor.to('cpu').state_dict(), path+"DDPG_actor.pth")
        self.actor.to(self.device)

        torch.save(self.critic.to('cpu').state_dict(), path+"DDPG_critic.pth")
        self.critic.to(self.device)

    def load(self, path="./"):
        self.actor.load_state_dict(torch.load(path+"DDPG_actor.pth"))
        self.critic.load_state_dict(torch.load(path+"DDPG_critic.pth"))

