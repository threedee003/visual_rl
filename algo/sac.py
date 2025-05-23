import os
import numpy as np
import torch
import torch.nn as nn
from models.actor_critic import Actor, QNet


class SAC:
    '''
    Soft Actor Critic

    Original Paper : Soft Actor-Critic: Off-Policy Maximum ENtropy Deep Reinforcement Learning With a Stochastic Actor
    Link : https://arxiv.org/pdf/1801.01290
    '''
    def __init__(self,
                 img_size: int,
                 feature_dim: int,
                 action_dim: int,
                 gamma: float,
                 tau: float,
                 alpha: float,
                 policy: str,
                 target_update_interval: int,
                 automatic_entropy_tuning: bool,
                 lr: float
    ) -> None:
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.action_space = (1, action_dim)
        self.pol_type = policy
        self.target_upd_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.critic = QNet().to(device=self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr = lr)
        self.critic_target = QNet().to(self.device)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        if self.pol_type == "Gaussian":
            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(self.action_space).to(self.device))
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)
            self.policy = Actor().to(self.device)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr)



    def select_action(self, state, eval = False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(torch.float16).to(self.device)[None]
        if not eval:
            action, _, _ = self.policy.act(state)
        else:
            _, _, action = self.policy.act(state)
        return action.detach().cpu().numpy()
    

    def update_params(self, buff, batch_size, updates):
        state_b, action_b, reward_b, next_state_b, mask_b = buff.sample_buffer(batch_size = batch_size)

        state_b = torch.from_numpy(state_b).to(torch.float16).to(self.device)
        next_state_b = torch.from_numpy(next_state_b).to(torch.float16).to(self.device)
        action_b = torch.from_numpy(action_b).to(torch.float16).to(self.device)
        reward_b = torch.from_numpy(reward_b).to(torch.float16).to(self.device)
        mask_b = torch.from_numpy(mask_b).to(torch.float16).to(self.device)

        with torch.no_grad():
            

