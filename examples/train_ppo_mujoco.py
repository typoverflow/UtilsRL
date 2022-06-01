import os
import sys
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operator import itemgetter

from UtilsRL.math import discounted_cum_sum
from UtilsRL.rl.normalizer import RunningNormalizer
from UtilsRL.rl.buffer import TransitionReplayPool
from UtilsRL.rl.actor import SquashedGaussianActor
from UtilsRL.rl.critic import SingleCritic
from UtilsRL.rl.net import MLP
from UtilsRL.logger import TensorboardLogger
from UtilsRL.monitor import Monitor
from UtilsRL.exp import parse_args, setup

# 1. Set up logger and arguments
args = parse_args("./examples/configs/ppo_mujoco.py")
logger = TensorboardLogger(args.log_path, args.name)
setup(args, logger)

# 2. Add environment specs to arguments
task = "Hopper-v3"
env = gym.make(task)
args["obs_space"] = env.observation_space
args["action_space"] = env.action_space
args["obs_shape"] = env.observation_space.shape[0]
args["action_shape"] = env.action_space.shape[0]

# 3. Define structures of networks
actor_backend = MLP(args.obs_shape, 0, args.actor_hidden_dims, activation=nn.Tanh, device=args.device)
critic1_backend = MLP(args.obs_shape, 0, args.critic_hidden_dims, activation=nn.Tanh, device=args.device)
# critic2_backend = MLP(args.obs_shape, 0, args.critic_hidden_dims, activation=nn.Tanh, device=args.device)

actor = SquashedGaussianActor(
    actor_backend, args.actor_hidden_dims[-1], args.action_shape, 
    device=args.device, reparameterize=False, conditioned_logstd=True, hidden_dims=args.actor_output_hidden_dims
).to(args.device)
critic1 = SingleCritic(
    critic1_backend, args.critic_hidden_dims[-1], 1, 
    device=args.device, hidden_dims=args.critic_output_hidden_dims
).to(args.device)
# critic2 = SingleCritic(
#     critic2_backend, args.critic_hidden_dims[-1], 1, 
#     device=args.device
# ).to(args.device)

actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
critic_optim = torch.optim.Adam([*critic1.parameters()], lr=args.critic_lr)

obs_normalizer = RunningNormalizer(shape=args.obs_shape)

# 4. Define the agent logic, including update, get_action and get_value
def update(data_batch):
    obs_batch, action_batch, logprob_batch, advantage_batch, return_batch = \
        itemgetter("obs", "action", "logprob", "advantage", "return")(data_batch)
    obs_batch = torch.from_numpy(obs_batch).float().to(args.device)
    action_batch = torch.from_numpy(action_batch).float().to(args.device)
    logprob_batch = torch.from_numpy(logprob_batch).float().to(args.device)
    advantage_batch = torch.from_numpy(advantage_batch).float().to(args.device)
    return_batch = torch.from_numpy(return_batch).float().to(args.device)
        
    actor_loss_value = critic_loss_value = entropy_loss_value = tot_approx_kl = clip_fraction = 0
    for actor_step in range(args.actor_repeat_step):
        new_logprob, new_entropy = actor.evaluate(obs_batch, action_batch)
        ratio = torch.exp(new_logprob - logprob_batch)
        approx_kl = (logprob_batch - new_logprob).mean().item()
        if approx_kl > 1.5 * args.target_kl:
            break
        
        surr1 = advantage_batch * ratio
        surr2 = advantage_batch * torch.clip(ratio, 1-args.clip_range, 1+args.clip_range)
        actor_loss = - (torch.min(surr1, surr2)).mean()
        entropy_loss = - new_entropy.mean()

        actor_optim.zero_grad()
        (actor_loss + args.entropy_coeff*entropy_loss).backward()
        actor_optim.step()
        
        actor_loss_value += actor_loss.detach().cpu().item()
        entropy_loss_value += entropy_loss.detach().cpu().item()
        tot_approx_kl += approx_kl
        clip_fraction += (abs(ratio-1.0) > args.clip_range).float().mean().detach().cpu().item()
        
    for critic_step in range(args.critic_repeat_step):
        value = critic1(obs_batch)
        critic_loss = F.mse_loss(value, return_batch)
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()
        
        critic_loss_value += critic_loss.detach().cpu().item()
        
    ret_dict = {
        "loss/critic": critic_loss/critic_step, 
        "misc/actor_update": actor_step
    }
    if actor_step > 0:
        ret_dict.update({
            "loss/actor": actor_loss_value/actor_step,
            "loss/entropy": entropy_loss_value/actor_step,
            "misc/approx_kl": tot_approx_kl/actor_step,
            "misc/clip_fraction": clip_fraction/actor_step
        })
    return ret_dict

@torch.no_grad()
def get_value(obs):
    if not isinstance(obs, torch.Tensor):
        obs = torch.from_numpy(obs).float().to(args.device)
    if len(obs.shape) == 1:
        obs = torch.unsqueeze(obs, 0)
    return torch.squeeze(critic1(obs)).detach().cpu().numpy()

@torch.no_grad()
def get_action(obs, deterministic=False):
    if not isinstance(obs, torch.Tensor):
        obs = torch.from_numpy(obs).float().to(args.device)
    if len(obs.shape) == 1:
        obs = torch.unsqueeze(obs, 0)
    action, logprob, _ = actor.sample(obs, deterministic=deterministic, return_mean_logstd=False)
    return torch.squeeze(action).cpu().numpy(), torch.squeeze(logprob).cpu().numpy() if logprob else logprob


# 5. Define the training logics
def compute_gae(rewards, values, last_v, gamma=0.99, lam=0.95):
    rewards = np.append(rewards, last_v)
    values = np.append(values, last_v)
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    # 检查计算delta的过程是否会导致reward之类的发生变化
    gae = discounted_cum_sum(deltas, gamma * lam)
    ret = gae + values[:-1]
    # gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    return gae, ret
    
buffer = TransitionReplayPool(args.obs_space, args.action_space, args.buffer_size, extra_fields={
    "advantage": {"shape": (1, ), "dtype": "float"}, 
    "logprob": {"shape": (1, ), "dtype": "float"}, 
    "return": {"shape": (1, ), "dtype": "float"}, 
    "value": {"shape": (1, ), "dtype": "float"}
})

tot_env_step = 0
# traj_length = traj_return = 0
env = gym.make(task)
for i_epoch in Monitor("PPO Training").listen(range(args.max_epoch)):
    buffer.clear()
    obs, done = env.reset(), False
    sample_ph = buffer.get_placeholder(args.sample_per_epoch)
    traj_length = traj_return = traj_start = 0
    for env_step in range(args.sample_per_epoch):
        action, logprob = get_action(obs)
        next_obs, reward, done, _ = env.step(action)
        # traj_return += reward
        traj_length += 1
        tot_env_step += 1
        
        value = get_value(obs)
        
        sample_ph["obs"][env_step] = obs
        sample_ph["action"][env_step] = action
        sample_ph["logprob"][env_step] = logprob
        sample_ph["next_obs"][env_step] = next_obs
        sample_ph["reward"][env_step] = reward / args.reward_scale
        sample_ph["done"][env_step] = done
        sample_ph["value"][env_step] = value
        
        epoch_ended = env_step == args.sample_per_epoch - 1
        timeout = traj_length == args.max_traj_length
        
        if done or timeout or epoch_ended:
            if timeout or epoch_ended:
                last_v = get_value(next_obs)
            else:
                last_v = 0
            gae, ret = compute_gae(sample_ph["reward"][traj_start:env_step+1], sample_ph["value"][traj_start:env_step+1], last_v=last_v)
            sample_ph["return"][traj_start:env_step+1] = ret.reshape(-1, 1)
            sample_ph["advantage"][traj_start:env_step+1] = gae.reshape(-1, 1)
            # for field in buffer.field_names:
                # sample_ph[field] = sample_ph[field][:traj_length]
            # buffer.add_samples(sample_ph)
            
            next_obs, done = env.reset(), False
            traj_length = 0
            traj_start = env_step + 1
        
        obs = next_obs
    
    buffer.add_samples(sample_ph)
    data_batch = buffer.random_batch(args.batch_size)
    train_loss = update(data_batch)
    
    if i_epoch % args.eval_interval == 0:
        traj_lengths = []
        traj_returns = []
        for traj_id in range(args.eval_num_traj):
            traj_return = traj_length = 0
            state, done = env.reset(), False
            for step in range(args.max_traj_length):
                action, _= get_action(state, deterministic=True)
                next_state, reward, done, _ = env.step(action)
                traj_return += reward
                traj_length += 1
                if done:
                    break
            traj_lengths.append(traj_length)
            traj_returns.append(traj_return)
        train_loss.update({
            "eval/traj_return": np.mean(traj_returns), 
            "eval/traj_length": np.mean(traj_lengths)
        })
    
    logger.log_scalars("", train_loss, step=i_epoch)  
        