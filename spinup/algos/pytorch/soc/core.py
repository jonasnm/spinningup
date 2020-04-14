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
        # torch.squeeze(q, -1)  # Critical to ensure q has right shape.
        return q.gather(-1, option).squeeze(-1)


class SquashedGaussianSOCActor(nn.Module):

    def __init__(self, obs_dim, act_dim, N_options, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], N_options*act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], N_options*act_dim)
        self.act_limit = act_limit
        self.currOption = np.array(0, dtype=np.long)

    def forward(self, obs, options, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)

        # Get intra-option policy parameters corresponding to the given option
        #mu = mu.gather(-1, options)
        #log_std = log_std.gather(-1, options)

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
            logp_pi = pi_distribution.log_prob(pi_action)  # .sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action -
                           F.softplus(-2*pi_action)))  # .sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        # self.act_limit * pi_action #TODO: Change action-space for my env instead. Though OG seems wrong - does not account for the sign of pi_action, e.g. a in [0,500] to range 500*[-1,1] = [-500,500]
        pi_action = pi_action.gather(-1, options)

        return pi_action, logp_pi


class QwFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, N_options, hidden_sizes, activation):
        super().__init__()
        self.z = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.Qw = nn.Linear(hidden_sizes[-1], N_options)
        self.beta = nn.Sequential(
            nn.Linear(
                hidden_sizes[-1], N_options),
            nn.Sigmoid())

    def forward(self, obs):
        z = self.z(obs)
        Qw = self.Qw(z)
        beta = self.beta(z)
        return Qw, beta


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
        self.q1 = MLPQuFunction(
            obs_dim, act_dim, N_options, hidden_sizes, activation)
        self.q2 = MLPQuFunction(
            obs_dim, act_dim, N_options, hidden_sizes, activation)
        self.Qw = QwFunction(obs_dim, act_dim, N_options,
                             hidden_sizes, activation)

    def act(self, obs, w=None, deterministic=False):
        if w is None:
            w = self.pi.currOption
        with torch.no_grad():
            w = torch.as_tensor(w, dtype=torch.long)
            a, _ = self.pi(obs, w, deterministic, False)
            #TODO: getOption(obs, w, deterministic)
            return a.numpy()

    def getOption(self, obs):
        w = self.pi.currOption
        obs = torch.as_tensor(obs, dtype=torch.float32)
        Qw, beta = self.Qw(obs)
        # keep current option with probability 1-beta_w
        if (1-beta[w]) > np.random.rand():
            option = w

        # else get new option
        else:
            N_options = len(beta)
            if np.random.rand() > self.eps:
                option = np.argmax(Qw[w].detach().numpy())
            else:
                option = np.random.choice(np.arange(N_options))
        self.pi.currOption = option
        return option
