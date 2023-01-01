# %%
# load dependencies
import os
import sys
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from operator import itemgetter

# %%
# 1. Set up logger and experiments
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import DummyLogger, TensorboardLogger
args = parse_args("./examples/configs/rainbow_atari.py")
if args.debug:
    logger = DummyLogger()
else:
    logger = TensorboardLogger(args.log_path, "_".join([args.name, args.task]))
setup(args, logger, args.device)
print(args)


# %%
# 2. Add environment specs to arguments
from UtilsRL.env.atari import wrap_deepmind
task = args.task
env = wrap_deepmind(task, episode_life=True, clip_rewards=True)
eval_env = wrap_deepmind(task, episode_life=False, clip_rewards=False, render_mode="rgb_array")
args["observation_space"] = env.observation_space
args["action_space"] = env.action_space
np_ftype, torch_ftype = args.UtilsRL.numpy_fp, args.UtilsRL.torch_fp


# %%
# 3. Define the agent and buffer
from UtilsRL.rl.buffer import PrioritizedSimpleReplay, PrioritizedFlexReplay, convert_space_to_spec
from UtilsRL.net.cnn import CNN, conv2d_out_size
from UtilsRL.net.basic import NoisyLinear
from UtilsRL.net.utils import reset_noise_layer
from UtilsRL.rl.critic import C51DQN
from UtilsRL.misc.decorator import profile
from UtilsRL.rl.video_recorder import VideoRecorder
field_specs = {
    "obs": convert_space_to_spec(args.observation_space), 
    "action": convert_space_to_spec(args.action_space), 
    "next_obs": convert_space_to_spec(args.observation_space), 
    "reward": {"shape": [1, ], "dtype": np.float32}, 
    "done": {"shape": [1, ], "dtype": np.float32}, 
}
alpha = args.alpha if args.use_per else 0
buffer = PrioritizedFlexReplay(args.buffer_size, field_specs, "proportional", alpha, cache_max_size=2000)

class RainbowAgent():
    def __init__(self, args):
        self.device = args.device
        self.target_update_interval = args.target_update_interval
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.prior_eps = args.prior_eps
        self.video_recorder = VideoRecorder(args.log_path)
        # self.traj_limit = args.max_episode_length
        
        # epsilon greedy
        self.epsilon = args.max_epsilon
        self.max_epsilon = args.max_epsilon
        self.min_epsilon = args.min_epsilon
        self.epsilon_decay = args.epsilon_decay

        # beta 
        self.beta = args.min_beta
        self.max_beta = args.max_beta
        self.min_beta = args.min_beta
        self.beta_decay = args.beta_decay
        
        self.use_categorical = args.use_categorical
        self.use_double = args.use_double
        self.use_noisy = args.use_noisy
        self.use_per = args.use_per
        self.use_n_step = args.use_n_step
        self.n_step = 1 if not self.use_n_step else args.n_step
        
        # for DQN
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.atom_size = args.atom_size
        dqn_backend = CNN(input_channel=args.frame_stack*(3 if args.use_rgb else 1), channels=args.channels, kernels = args.kernels, strides=args.strides)
        linear_size = 84
        for k, s in zip(args.kernels, args.strides):
            linear_size = conv2d_out_size(linear_size, k, s)
        linear_size = linear_size*linear_size * args.channels[-1]
        self.dqn = C51DQN(backend=dqn_backend, 
                          input_dim=linear_size, 
                          output_dim_adv=args.action_space.n, 
                          output_dim_value=1, 
                          num_atoms=args.atom_size, 
                          v_min=args.v_min, 
                          v_max=args.v_max, 
                          hidden_dims=args.hidden_dims, 
                          linear_layer=NoisyLinear if args.use_noisy else nn.Linear
                          ).to(args.device)
        if self.use_double:
            dqn_target_backend = CNN(input_channel=args.frame_stack*(3 if args.use_rgb else 1), channels=args.channels, kernels = args.kernels, strides=args.strides)
            self.dqn_target = C51DQN(backend=dqn_target_backend, 
                                input_dim=linear_size, 
                                output_dim_adv=args.action_space.n, 
                                output_dim_value=1, 
                                num_atoms=args.atom_size, 
                                v_min=args.v_min, 
                                v_max=args.v_max, 
                                hidden_dims=args.hidden_dims, 
                                linear_layer=NoisyLinear if args.use_noisy else nn.Linear
                                ).to(args.device)
            self.dqn_target.load_state_dict(self.dqn.state_dict())
        else:
            self.dqn_target = self.dqn
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=args.lr, eps=1.5e-4)
        
    @torch.no_grad()
    def get_action(self, state: np.ndarray, deterministic=False):
        if not deterministic and self.epsilon > np.random.random():
            selected_action = args.action_space.sample()
        else:
            state = np.asarray(state)
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device).unsqueeze(0)).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action
    
    def update_epsilon(self, step):
        self.epsilon = max(self.max_epsilon - self.epsilon_decay * step, self.min_epsilon)
        
    def update_beta(self, step):
        self.beta = min(self.min_beta + self.beta_decay * step, self.max_beta)
    
    def rainbow_update(self, batch_data, batch_is):
        batch_is = torch.from_numpy(batch_is).float().to(self.device)
        batch_obs = torch.from_numpy(batch_data["obs"]).float().to(self.device)
        batch_next_obs = torch.from_numpy(batch_data["next_obs"]).float().to(self.device)
        batch_action = torch.from_numpy(batch_data["action"]).to(self.device)
        batch_reward = torch.from_numpy(batch_data["reward"]).float().to(self.device)
        batch_done = torch.from_numpy(batch_data["done"]).float().to(self.device)
        
        if self.use_categorical:
            delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
            with torch.no_grad():
                next_action = self.dqn(batch_next_obs).argmax(-1)
                next_dist = self.dqn_target.dist(batch_next_obs)
                next_dist = next_dist[range(self.batch_size), next_action]
                
                target_z = batch_reward + (1-batch_done) * (self.gamma**self.n_step) * self.dqn.support
                target_z = target_z.clamp(min=self.v_min, max=self.v_max)
                proj_dist = (
                    1 - (target_z.unsqueeze(1) - self.dqn.support.view(1, -1, 1)).abs() / delta_z
                ).clamp(0, 1) * next_dist.unsqueeze(1)
                proj_dist = proj_dist.sum(-1)
            dist = self.dqn.dist(batch_obs)
            log_p = torch.log(dist[range(self.batch_size), batch_action.squeeze()])
            loss = -(proj_dist * log_p).sum(1)
        else:
            q_value = self.dqn(batch_obs).gather(1, batch_action.squeeze())
            with torch.no_grad():
                next_q_value = self.dqn_target(batch_next_obs).gather(1, self.dqn(batch_next_obs).argmax(dim=1, keepdim=True)[0])
            target = batch_reward + (self.gamma ** self.n_step)*(1-batch_done)*next_q_value
            loss = F.mse_loss(q_value, target, reduction="none")
        
        weighted_loss = torch.mean(loss*batch_is)
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # for per update
        loss_for_prior = loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        
        if self.use_noisy:
            reset_noise_layer(self.dqn)
            reset_noise_layer(self.dqn_target)
            
        return {"weighted_loss": weighted_loss.detach().cpu().item()}, new_priorities

    def target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())
    
    def eval_on_env(self, env, eval_num):
        self.dqn.eval()
        traj_returns = []
        traj_lengths = []
        traj_return = 0
        traj_length = 0
        self.video_recorder.reset()
        # traj_limit = self.traj_limit
        for i in range(eval_num):
            obs, done = env.reset(), False
            traj_return = traj_length = 0
            while True:
                obs, reward, done, metadata = env.step(self.get_action(obs, deterministic=True))
                traj_return += reward
                traj_length += 1
                if i == eval_num-1:
                    self.video_recorder.record(metadata["rgb"])
                if done:
                    traj_returns.append(traj_return)
                    traj_lengths.append(traj_length)
                    break
        self.video_recorder.save(args.name+"_"+args.task, "mp4")
        self.dqn.train()
        return np.mean(traj_returns), np.mean(traj_lengths)

agent = RainbowAgent(args)

# %%
# 4. define the loop
import copy
from collections import deque
from UtilsRL.monitor import Monitor
batch_size = args.batch_size
n_step, use_n_step = args.n_step, args.use_n_step
# traj_limit = args.max_episode_length
logger: TensorboardLogger = args.logger
tot_env_step = 0
tot_agent_step = 0
traj_return = 0
traj_length = 0
best_score = -np.inf
best_model = None
obs, done = env.reset(), False
buffer.reset()
reward_buffer = deque(maxlen=n_step)
for frame_idx in Monitor("train").listen(range(1, args.num_frames+1)):
    action = agent.get_action(obs)
    next_obs, reward, done, _ = env.step(action)
    # traj_return += reward
    traj_length += 1

    buffer.add_sample({
        "obs": obs, 
        "action": action, 
    })
    if use_n_step:
        reward_buffer.append(reward)
        if len(reward_buffer) == n_step:
            buffer.add_sample({
                "next_obs": next_obs, 
                "reward": sum(reward_buffer), 
                "done": done
            })
            # buffer.commit(1)
    
    agent.update_beta(frame_idx)
    agent.update_epsilon(frame_idx)
    
    if done:
        while len(reward_buffer) > 1:
            reward_buffer.popleft()
            buffer.add_sample({
                "next_obs": next_obs, 
                "reward": sum(reward_buffer), 
                "done": done
            })
        reward_buffer.popleft()
        next_obs = env.reset()
        traj_length = 0
        
    obs = next_obs
    
    if frame_idx < args.warmup_frame:
        buffer.commit()
        continue
    
    if frame_idx % args.update_every == 0:
        buffer.commit()   # lazy commit the samples
        batch_data, batch_is, batch_idx = buffer.random_batch(batch_size, beta=agent.beta)
        train_loss, batch_metric = agent.rainbow_update(batch_data, batch_is)
        buffer.batch_update(batch_idx, batch_metric)
        tot_agent_step += 1
        if args.use_double and tot_agent_step % args.target_update_interval == 0:
            agent.target_hard_update()
    
    if frame_idx % args.log_interval == 0:
        logger.log_scalars("Train", train_loss, step=frame_idx)
        logger.log_scalars("misc", {
            "beta": agent.beta, 
            "epsilon": agent.epsilon
        }, step=frame_idx)
        
    if frame_idx % args.eval_interval == 0:
        score, length = agent.eval_on_env(eval_env, eval_num=args.eval_num)
        if score > best_score:
            best_model = copy.deepcopy(agent.dqn.state_dict())
            best_score = score
        logger.log_scalars("Eval", {
            "score": score, 
            "length": length
        }, step=frame_idx)
    
