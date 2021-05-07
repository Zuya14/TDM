from DDPG import DDPG
from HindsightReplayBuffer import HindsightReplayBuffer


class GC_DDPG(DDPG):

    def __init__(self, state_size, action_size, goal_size, hidden_size=256, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
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

        self.buffer = HindsightReplayBuffer(
            buffer_size=replay_size,
            state_size=state_size,
            action_size=action_size,
            goal_size=goal_size,
            device=device,
        )


        # Actor-Criticのネットワークを構築する．
        self.actor = ActorNetwork(
            state_size=state_size+goal_size,
            action_size=action_size,
            hidden_size=hidden_size
        ).to(device)
        self.critic = CriticNetwork(
            state_size=state_size+goal_size,
            action_size=action_size,
            hidden_size=hidden_size
        ).to(device)
        self.critic_target = CriticNetwork(
            state_size=state_size+goal_size,
            action_size=action_size,
            hidden_size=hidden_size
        ).to(device).eval()

        # ターゲットネットワークの重みを初期化し，勾配計算を無効にする．
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # オプティマイザ．
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

state_t と　state_t+1 のそれぞれに対して goal を concat するのが下位層だと面倒なので上位層（Bufferくらい）でconcatしたい

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''