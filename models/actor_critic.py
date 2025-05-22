from models.resnets import ResNet

import os
import torch
import torch.nn as nn
import torch.distributions.distribution


def _get_activation_(activation):
    if activation == 'relu':
        return nn.ReLU
    elif activation == 'tanh':
        return nn.Tanh
    else:
        raise NotImplementedError("only 'relu' and 'tanh' are supported")


class QNet(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 feature_dim: int = 64,
                 action_dim: int = 7,
                 mean_std_samp: bool = False,
                 activation: str = 'relu',
                 checkpoint_dir: str = './weights',
                 log_std_max: float = 20,
                 log_std_min: float = -2,
                 action_space: tuple  = None,
                 obs_type: str = "image",
                 obs_dim: int = None
    ) -> None:
        super(QNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs_type = obs_type
        if self.obs_type == 'image':
            self.resnet = ResNet()
            self.feature_dim = feature_dim
        else:
            self.feature_dim = obs_dim
        self.activation = _get_activation_(activation)
        self.q1 = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            self.activation,
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            self.activation,
            nn.Linear(self.feature_dim // 4, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            self.activation,
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            self.activation,
            nn.Linear(self.feature_dim // 4, 1)
        )
        self.check_pt_file = os.path.join(checkpoint_dir, "actor_sac")
        self.apply(self._weights_init_)

    def forward(self, x):
        if self.obs_type == 'image':
            x = self.resnet(x)
        x1 = self.q1(x)
        x2 = self.q2(x)
        return x1, x2


    def save_checkpoint(self):
        torch.save(self.state_dict(), self.check_pt_file)

    def load_checkpt(self):
        self.load_state_dict(torch.load(self.check_pt_file))


    def _weights_init_(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain = 1)
            torch.nn.init.constant_(m.bias, 0)




class Actor(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 feature_dim: int = 64,
                 action_dim: int = 7,
                 mean_std_samp: bool = False,
                 activation: str = 'relu',
                 checkpoint_dir: str = './weights',
                 log_std_max: float = 20,
                 log_std_min: float = -2,
                 action_space: tuple  = None,
                 obs_type: str = "image",
                 obs_dim: int = None
    ) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs_type = obs_type
        if self.obs_type == 'image':
            self.resnet = ResNet()
            self.feature_dim = feature_dim
        else:
            self.feature_dim = obs_dim


        self.activation = _get_activation_(activation)
        self.layers = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            self.activation,
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            self.activation,
        )
        self.mean_layer = nn.Linear(self.feature_dim // 4, action_dim)
        self.log_std_layer = nn.Linear(self.feature_dim // 4, action_dim)
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.check_pt_file = os.path.join(checkpoint_dir, "actor_sac")
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space[1] - action_space[0]) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space[1] + action_space[0]) / 2.)
        self.eps = 1e-6
        self.apply(self._weights_init_)

    def forward(self, x):
        if self.obs_type == 'image':
            x = self.resnet(x)
        x = self.layers(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, max=self.log_std_max, min=self.log_std_min)
        return mean, log_std
    
    def act(self, x):
        mean, log_std = self.act(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean , std)
        xt = normal.rsample()
        yt = torch.tanh(xt)
        action = yt * self.action_scale + self.action_bias
        log_prob = normal.log_prob(xt)
        log_prob -= torch.log(self.action_scale * (1 - yt.pow(2)) + self.eps)
        log_prob = torch.sum(log_prob, dim = 1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale +  self.action_bias
        return action, log_prob, mean

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.check_pt_file)

    def load_checkpt(self):
        self.load_state_dict(torch.load(self.check_pt_file))

    def _weights_init_(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain = 1)
            torch.nn.init.constant_(m.bias, 0)
