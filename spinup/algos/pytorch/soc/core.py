import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


# Christian Jenssen
class MLPQuFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, N_options, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] +
                     list(hidden_sizes) + [N_options], activation)

    def forward(self, obs, option, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return q.gather(-1, option).squeeze(-1)


class SquashedGaussianSOCActor(nn.Module):

    def __init__(self, obs_dim, act_dim, N_options, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], N_options*act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], N_options*act_dim)
        self.beta = nn.Sequential(
            nn.Linear(
                hidden_sizes[-1], N_options),
            nn.Sigmoid())
        self.act_limit = act_limit
        self.act_dim = act_dim
        self.currOption = np.array(0, dtype=np.long)
        self.N_options = N_options

    def getBeta(self, obs):
        net_out = self.net(obs)
        beta = self.beta(net_out)
        return beta

    def forward(self, obs, options, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        z_mu = self.mu_layer(net_out)
        z_log_std = self.log_std_layer(net_out)
        mu = z_mu.view(-1, self.act_dim, self.N_options)
        log_std = z_log_std.view(-1, self.act_dim, self.N_options)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)

            # get log-probs, sum over action dimension
            logp_pi = pi_distribution.log_prob(pi_action)
            logp_pi -= (2*(np.log(2) - pi_action -
                           F.softplus(-2*pi_action)))
            logp_pi = logp_pi.sum(dim=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def selectOptionAct(self, options, pi_action):
        options = options.repeat(1, self.act_dim).view(-1, self.act_dim, 1)
        pi_action = pi_action.gather(-1, options).squeeze(-1)
        return pi_action


class QwFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, N_options, hidden_sizes, activation):
        super().__init__()
        self.z = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.Qw = nn.Linear(hidden_sizes[-1], N_options)

    def forward(self, obs):
        z = self.z(obs)
        Qw = self.Qw(z)
        return Qw


class MLPOptionCritic(nn.Module):

    def __init__(self, observation_space, action_space, N_options, hidden_sizes=(256, 256),
                 activation=nn.ReLU, eps=0.1):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.eps = eps

        # build policy and value functions
        self.pi = SquashedGaussianSOCActor(
            obs_dim, act_dim, N_options, hidden_sizes, activation, act_limit)
        self.q = MLPQuFunction(
            obs_dim, act_dim, N_options, hidden_sizes, activation)
        self.Qw = QwFunction(obs_dim, act_dim, N_options,
                             hidden_sizes, activation)

    def act(self, obs, w=None, deterministic=False):
        if w is None:
            w = self.pi.currOption
        with torch.no_grad():
            w = torch.as_tensor(w, dtype=torch.long)
            a, _ = self.pi(obs, w, deterministic, False)
            a = self.pi.selectOptionAct(w, a)
            return a[0].numpy()

    def getOption(self, obs, ChangeOption=True):
        w = self.pi.currOption
        obs = torch.as_tensor(obs, dtype=torch.float32)
        beta = self.pi.getBeta(obs)

        if ChangeOption:
            # keep current option with probability 1-beta_w
            if (1-beta[w]) > np.random.rand():
                option = w

            # else get new option
            else:
                Qw = self.Qw(obs)
                probs = torch.softmax(Qw/0.1, dim=-1).detach()
                pi_O = torch.distributions.Categorical(probs)
                option = pi_O.sample().item()
                self.pi.currOption = option
        else:
            Qw = self.Qw(obs)
            probs = torch.softmax(Qw/0.1, dim=-1).detach()
            return probs
