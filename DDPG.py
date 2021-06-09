import torch
import torch.nn as nn
import torch.nn.functional as F
from ReplayBuffer import ReplayBuffer
import random

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

# actorのネットワーク
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()

        # self.net = nn.Sequential(
        #     nn.Linear(state_size, hidden_size),
        #     nn.ELU(inplace=True),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ELU(inplace=True),
        #     nn.Linear(hidden_size, action_size),
        # )

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=True),
        )

        init_w=3e-3
        self.last_fc = nn.Linear(hidden_size, action_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, states, return_preactivations=False):

        preactivation = self.last_fc(self.net(states))
        output = torch.tanh(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output
        # return torch.tanh(self.net(states))

    # def __init__(self, state_size, action_size, hidden_size=64):
    #     super().__init__()
    #     self.num_hidden_layers = hidden_size
    #     self.input = state_size
    #     self.output_action = action_size
    #     self.init_w = 3e-3

    #     #Dense Block
    #     self.dense_1 = nn.Linear(self.input, self.num_hidden_layers)
    #     self.relu1 = nn.ReLU(inplace=True)
    #     self.dense_2 = nn.Linear(self.num_hidden_layers, self.num_hidden_layers)
    #     self.relu2 = nn.ReLU(inplace=True)
    #     self.output = nn.Linear(self.num_hidden_layers, self.output_action)
    #     self.tanh = nn.Tanh()

    # def init_weights(self, init_w):
    #     self.dense_1.weight.data = fanin_init(self.dense_1.weight.data.size())
    #     self.dense_2.weight.data = fanin_init(self.dense_2.weight.data.size())
    #     self.output.weight.data.uniform_(-init_w, init_w)

    # def forward(self, states):
    #     x = self.dense_1(states)
    #     x = self.relu1(x)
    #     x = self.dense_2(x)
    #     x = self.relu2(x)
    #     output = self.output(x)
    #     output = self.tanh(output)
    #     return output

class CNet(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()

        self.fc1     = nn.Linear(state_size + action_size, hidden_size)
        self.fc2     = nn.Linear(hidden_size, hidden_size)
        self.fc3_adv = nn.Linear(hidden_size, action_size)
        self.fc3_v   = nn.Linear(hidden_size, 1)

        self.fn = nn.ELU(inplace=True)

    def forward(self, x):
        h1 = self.fn(self.fc1(x))
        h2 = self.fn(self.fc2(h1))
        adv = self.fc3_adv(h2)
        val = self.fc3_v(h2).expand(-1, adv.size(1))
        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return output

# criticのネットワーク（状態と行動を入力にしてQ値を出力）
class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, 1),
        )
        # self.net1 = CNet(state_size, action_size, hidden_size)
        # self.net2 = CNet(state_size, action_size, hidden_size)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net1(x), self.net2(x)

class DDPG:

    def __init__(self, state_size, action_size, hidden_size=64, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 batch_size=256, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3,
                 replay_size=10**6, start_steps=10**4, tau=5e-3, alpha=0.2, reward_scale=1.0, epsilon_decay = 50000):

        super().__init__()

        self.name = 'DDPG'

        # リプレイバッファ．
        self.buffer = ReplayBuffer(
            buffer_size=replay_size,
            state_size=state_size,
            action_size=action_size,
            device=device,
        )
        print(state_size, action_size)
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

