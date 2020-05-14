import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


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


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)


class MLPOptionCritic(nn.Module):
    def __init__(self, observation_space, action_space, N_options,
                 hidden_sizes=(64, 64), activation=nn.Tanh, eps=0.1):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.eps = eps

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = GaussianOCActor(
                obs_dim, act_dim, N_options, hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.Qw = QwFunction(obs_dim, act_dim, N_options,
                             hidden_sizes, activation)

    def step(self, obs):
        w = self.pi.currOption
        with torch.no_grad():
            a, logp_a = self.pi(self, obs, w)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs, w=None, deterministic=False):
        if w is None:
            w = self.pi.currOption
        with torch.no_grad():
            w = torch.as_tensor(w, dtype=torch.long)
            a, _ = self.pi(obs, w, deterministic, False)
            return a.numpy()

    # def act(self, obs):
    #     return self.step(obs)[0]
    def getOption(self, obs):
        w = self.pi.currOption
        obs = torch.as_tensor(obs, dtype=torch.float32)
        beta = self.pi.getBeta(obs)
        # keep current option with probability 1-beta_w
        if (1-beta[w]) > np.random.rand():
            option = w

        # else get new option
        else:
            N_options = len(beta)
            if np.random.rand() > self.eps:
                Qw = self.Qw(obs)
                option = np.argmax(Qw.detach().numpy())
            else:
                option = np.random.choice(np.arange(N_options))
        self.pi.currOption = option
        return option


class QwFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, N_options, hidden_sizes, activation):
        super().__init__()
        self.z = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.Qw = nn.Linear(hidden_sizes[-1], N_options)

    def forward(self, obs):
        z = self.z(torch.as_tensor(
            obs, dtype=torch.float32))
        Qw = self.Qw(z)
        return Qw


class GaussianOCActor(nn.Module):

    def __init__(self, obs_dim, act_dim, N_options, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], N_options*act_dim)

        log_std = -0.5 * np.ones(act_dim*N_options, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.log_std_layer = nn.Linear(hidden_sizes[-1], N_options*act_dim)
        self.beta = nn.Sequential(
            nn.Linear(
                hidden_sizes[-1], N_options),
            nn.Sigmoid())
        self.act_dim = act_dim
        self.currOption = np.array(0, dtype=np.long)
        self.N_options = N_options

    def getBeta(self, obs):
        net_out = self.net(torch.as_tensor(
            obs, dtype=torch.float32))
        beta = self.beta(net_out)
        return beta

    def forward(self, obs, options, deterministic=False, with_logprob=True):
        net_out = self.net(torch.as_tensor(
            obs, dtype=torch.float32))
        z_mu = self.mu_layer(net_out)
        #z_log_std = self.log_std_layer(net_out)
        z_log_std = self.log_std
        mu = z_mu.view(-1, self.act_dim, self.N_options)
        log_std = z_log_std.view(-1, self.act_dim, self.N_options)

        #log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian
            logp_pi = pi_distribution.log_prob(pi_action)

            # get log-probs, sum over action dimension, gather over selected options
            logp_pi = logp_pi.sum(dim=1)
            logp_pi = logp_pi.gather(-1, options.unsqueeze(-1)).squeeze(-1)
        else:
            logp_pi = None

        options = options.repeat(1, self.act_dim).view(-1, self.act_dim, 1)
        pi_action = pi_action.gather(-1, options).squeeze(0).squeeze(-1)

        return pi_action, logp_pi

    def _distribution(self, obs, w):
        net_out = self.net(torch.as_tensor(
            obs, dtype=torch.float32))
        z_mu = self.mu_layer(net_out)
        z_log_std = self.log_std_layer(net_out)
        #z_log_std = self.log_std
        mu = z_mu.view(-1, self.act_dim, self.N_options)
        log_std = z_log_std.view(-1, self.act_dim, self.N_options)

        #log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # select mu for given options
        w = w.repeat(1, self.act_dim).view(-1, self.act_dim, 1)
        mu = mu.gather(-1, w).squeeze(0).squeeze(-1)
        std = std.gather(-1, w).squeeze(0).squeeze(-1)

        return Normal(mu, std)
