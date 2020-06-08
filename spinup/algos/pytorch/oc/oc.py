import numpy as np
from copy import deepcopy
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.oc.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, N_options, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(
            size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.options_buf = np.zeros(size, dtype=np.long)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, obs2, options, done, val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = obs2
        self.options_buf[self.ptr] = options
        self.done_buf[self.ptr] = done

        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(
            deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, obs2=self.obs2_buf, options=self.options_buf, done=self.done_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def oc(env_fn, actor_critic=core.MLPOptionCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000, N_options=2, c=0.03, polyak=0.995,
        logger_kwargs=dict(), save_freq=10):
    """
    Vanilla Policy Gradient 

    (with GAE-Lambda for advantage estimation)

    Args:
        env_fn : A function which creates a copy of the environment.eg.add('lr', [1e-3, 2e-3, 3e-3])
        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to VPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space,
                      env.action_space, N_options, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.Qw])
    logger.log('\nNumber of parameters: \t pi: %d, \t Qw: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = VPGBuffer(obs_dim, act_dim, N_options,
                    local_steps_per_epoch, gamma, lam)

    # function for option- and action selection
    def get_action(o, deterministic=False):
        ac.getOption(o)
        return ac.act(torch.as_tensor(o, dtype=torch.float32),
                      deterministic=deterministic)

    # Set up function for computing critic loss

    def compute_loss_v(data):
        lossFun = torch.nn.MSELoss()
        o, o2, w, r, d = data['obs'], data['obs2'], data['options'], data['ret'], data['done']
        w = w.unsqueeze(-1)

        # Get action- and option-values
        Qw = ac.Qw(o)

        # Get option-value for the given options
        Qw = Qw.gather(-1, w).squeeze(-1)

        # Bellman backup for Q functions
        with torch.no_grad():

            # Termination probabilities
            beta_next = ac.pi.getBeta(o2)
            beta_next = beta_next.gather(-1, w).squeeze(-1)

            # Target Q-values and termination probability beta
            Qw_next = ac_targ.Qw(o2)
            V_next = Qw_next.max(1).values

            # select Qw and beta for given options, reduce to 1-dim tensor with squeeze
            Qw_next = Qw_next.gather(-1, w).squeeze(-1)

            gt = r + gamma * (1-d)*((1-beta_next) *
                                    Qw_next + beta_next*V_next)

        return lossFun(Qw, gt), gt

    # Set up function for computing intra-policy loss
    def compute_loss_pi(data, gt=None):
        o, o2, act, w, adv = data['obs'], data['obs2'], data['act'], data['options'], data['adv']
        w = w.unsqueeze(-1)

        # Policy loss
        pi = ac.pi._distribution(o, w)
        logp = pi.log_prob(act).sum(-1)
        loss_pi = -(logp*gt).mean()

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

        # Useful extra info
        ent = pi.entropy().mean().item()
        pi_info = dict(ent=ent)

        return loss_pi, loss_beta, pi_info

    # Set up optimizers for policy and value function
    Qw_optimizer = Adam(ac.Qw.parameters(), lr=vf_lr)
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    beta_optimizer = Adam(ac.pi.beta.parameters(), lr=pi_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()
        data['options'] = torch.as_tensor(
            data['options'], dtype=torch.long)

        # Value function learning
        for i in range(train_v_iters):
            Qw_optimizer.zero_grad()
            loss_Qw, gt = compute_loss_v(data)
            loss_Qw.backward()
            mpi_avg_grads(ac.Qw)    # average grads across MPI processes
            Qw_optimizer.step()

        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        beta_optimizer.zero_grad()
        loss_pi, loss_beta, pi_info = compute_loss_pi(data, gt)
        loss_pi.backward()
        loss_beta.backward()
        mpi_avg_grads(ac.pi)    # average grads across MPI processes
        pi_optimizer.step()
        beta_optimizer.step()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        # Log changes from update
        #ent = pi_info_old['ent']
        logger.store(LossPi=loss_pi, LossQw=loss_Qw)

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            # w = torch.as_tensor(ac.pi.currOption, dtype=torch.long)
            # a = ac.act(torch.as_tensor(
            #     o, dtype=torch.float32))
            a = get_action(o)
            Qw = ac.Qw(torch.as_tensor(
                o, dtype=torch.float32))
            w = torch.as_tensor(ac.pi.currOption, dtype=torch.long)
            Qw = Qw.gather(-1, w).squeeze(-1)

            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, o2, w, d, Qw)
            logger.store(Qw=Qw)

            # Update obs (critical!)
            o = o2

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' %
                          ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    3
                    # v = ac.v(torch.as_tensor(
                    #    o, dtype=torch.float32)).detach().numpy()
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

                # When reset state, select option with best option-value
                Qw = ac.Qw(torch.as_tensor(o, dtype=torch.float32))
                ac.pi.currOption = torch.argmax(Qw).numpy()

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform VPG update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='oc')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    oc(lambda: gym.make(args.env), actor_critic=core.MLPOptionCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
