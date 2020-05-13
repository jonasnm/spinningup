from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.soc.core as core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for soc agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(
            size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.options_buf = np.zeros(core.combined_shape(
            size, 1), dtype=np.long)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, option, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.options_buf[self.ptr] = option
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     option=self.options_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def soc(env_fn, actor_critic=core.MLPOptionCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1, N_options=2, eps=0.1):
    """
    Soft Option-Critic (SOC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SOC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, N_options,
                      **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module)
                       for module in [ac.pi, ac.Qw, ac.q])
    logger.log(
        '\nNumber of parameters: \t pi: %d, \t Qw: %d, \t Qu: %d\n' % var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, w, a, r, o2, d = data['obs'], data['option'], data['act'], data['rew'], data['obs2'], data['done']

        # Get action- and option-values
        Qu = ac.q(o, w, a)
        Qw = ac.Qw(o)

        # Get Qw and beta for the given options
        Qw = Qw.gather(-1, w).squeeze(-1)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions and corresponding log-probs come from *current* policy
            a2, logp_a2 = ac.pi(o2, w)

            # Termination probabilities
            beta_next = ac.pi.getBeta(o2)
            beta_next = beta_next.gather(-1, w).squeeze(-1)

            # Target Q-values and termination probability beta
            Qw_next = ac_targ.Qw(o2)
            Qw_next = Qw_next - alpha*logp_a2
            V_next = Qw_next.max(1).values

            # select Qw and beta for given options, reduce to 1-dim tensor with squeeze
            Qw_next = Qw_next.gather(-1, w).squeeze(-1)

            target = r + gamma * (1 - d) * ((1-beta_next) *
                                            Qw_next + beta_next*V_next)

        # MSE loss against Bellman backup
        loss_Qw = ((Qw - target)**2).mean()
        loss_q = ((Qu - target)**2).mean()

        # Useful info for logging
        q_info = dict(Qu=Qu.detach().numpy(),
                      Qw=Qw.detach().numpy(),
                      )

        return loss_Qw, loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o, o2, w = data['obs'], data['obs2'], data['option']
        pi_action, logp_pi = ac.pi(o, w)
        logp_pi = logp_pi.gather(-1, w).squeeze(-1)
        Qu_pi = ac.q(o, w, pi_action)

        # Entropy-regularized policy loss
        loss_pi = (alpha*logp_pi - Qu_pi).mean()

        # values for "beta-target"
        with torch.no_grad():
            Qw_next = ac.Qw(o2)
            V_next = Qw_next.max(-1).values
            Qw_next = Qw_next.gather(-1, w).squeeze(-1)
            Aw = (Qw_next - V_next) + c

        # Termination loss
        beta_next = ac.pi.getBeta(o2)
        beta_next = beta_next.gather(-1, w).squeeze(-1)
        loss_beta = (beta_next*Aw).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, loss_beta, pi_info

    # Set up optimizers for policy, q-functions
    q_optimizer = Adam(q_params, lr=lr)
    Qw_optimizer = Adam(ac.Qw.parameters(), lr=lr)
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    beta_optimizer = Adam(ac.pi.beta.parameters(), lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        Qw_optimizer.zero_grad()
        loss_Qw, loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        loss_Qw.backward()
        q_optimizer.step()
        Qw_optimizer.step()

        # Record things
        logger.store(LossQu=loss_q.item(), **q_info)
        logger.store(LossQw=loss_Qw.item())

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        beta_optimizer.zero_grad()
        loss_pi, loss_beta, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        loss_beta.backward()
        pi_optimizer.step()
        beta_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)
        logger.store(lossBeta=loss_beta.item())

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        ac.getOption(o)
        return ac.act(torch.as_tensor(o, dtype=torch.float32),
                      deterministic=deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    c = 0.07

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            w = ac.pi.currOption
            a = get_action(o)
        else:
            w = ac.pi.currOption
            a = get_action(o)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Deliberation cost
        r_tilde = r + (w == ac.pi.currOption)*c
        w = ac.pi.currOption

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, w, a, r_tilde, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0
            Qw = ac.Qw(torch.as_tensor(o, dtype=torch.float32))
            ac.pi.currOption = torch.argmax(Qw).numpy()

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                batch['option'] = torch.as_tensor(
                    batch['option'], dtype=torch.long)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Qu', with_min_and_max=True)
            logger.log_tabular('Qw', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQu', average_only=True)
            logger.log_tabular('LossQw', with_min_and_max=True)
            logger.log_tabular('lossBeta')
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='soc')
    parser.add_argument('--eps', type=float, default=0.1)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    soc(lambda: gym.make(args.env), actor_critic=core.MLPOptionCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, eps=args.eps)
