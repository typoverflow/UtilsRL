A Minimal Example of PPO with Continuous Control
================================================

This case study is based on the example `UtilsRL/examples/train_ppo_mujoco.py`, and it will brief \
you on the basic usage of UtilsRL. 

.. _Set up the logger and arguments:

1. Set up logger and arguments
------------------------------

We placed all training configurations in `UtilsRL/examples/configs/ppo_mujoco.py`. \
So simply parse the python configuration file with :func:`~UtilsRL.exp.parse_args` is enough.

.. code-block:: python

    args = parse_args("./examples/configs/ppo_mujoco.py")

Logger should be created according ``args.log_path`` and the name of this trial, so we create a :class:`~UtilsRL.logger.TensorboardLogger` object with 

.. code-block:: python

    logger = TensorboardLogger(args.log_path, args.name)

After this, setup the argument dict and register logger, device and seed by

.. code-block:: python

    setup(args, logger, args.device)

Simple and pretty. Everything about experiment management is done with 3 lines of code. 

2. Add environment info to args
-------------------------------

For convenience we hope to add environment information (such as action shape, obs shape and etc) to the argument. \
Here ``args`` is a :class:`~UtilsRL.misc.NameSpace` object, so both dict-like assignment and dot-like assignment are accepted. 

.. code-block:: python

    task = "Hopper-v3"
    env = gym.make(task)
    args["obs_space"] = env.observation_space
    args["action_space"] = env.action_space
    args["obs_shape"] = env.observation_space.shape[0]
    args["action_shape"] = env.action_space.shape[0]   

3. Define the structures of networks
------------------------------------

MLP backend is enough for mujoco environments, so we instantiate a :class:`~UtilsRL.rl.actor.SquashedGaussianActor` as the actor and a :class:`~UtilsRL.rl.critic.SingleCritic` as the critic. 

.. code-block:: python

    actor_backend = MLP(args.obs_shape, 0, args.actor_hidden_dims, activation=nn.Tanh, device=args.device)
    critic1_backend = MLP(args.obs_shape, 0, args.critic_hidden_dims, activation=nn.Tanh, device=args.device)
    # critic2_backend = MLP(args.obs_shape, 0, args.critic_hidden_dims, activation=nn.Tanh, device=args.device)

    actor = SquashedGaussianActor(
        actor_backend, args.actor_hidden_dims[-1], args.action_shape, 
        device=args.device, reparameterize=False, conditioned_logstd=False, hidden_dims=args.actor_output_hidden_dims
    ).to(args.device)
    critic1 = SingleCritic(
        critic1_backend, args.critic_hidden_dims[-1], 1, 
        device=args.device, hidden_dims=args.critic_output_hidden_dims
    ).to(args.device)

4. Define the actor udpate logic, action selection logic and training loops
---------------------------------------------------------------------------

This is more about `PPO Algorithms` itself, so we refer the readers to check the code in source file, and we only paste the \
training loop here in the doc. Note that observation data is transformed by a :class:`~UtilsRL.rl.normalizer.RunningNormalizer` before training. \
At the end of each epoch, the collected data are used to update the normalizer. 

.. code-block:: python

    for i_epoch in Monitor("PPO Training").listen(range(args.max_epoch)):
        buffer.clear()
        obs, done = env.reset(), False
        sample_ph = buffer.get_placeholder(args.sample_per_epoch)
        traj_length = traj_return = traj_start = 0
        for env_step in range(args.sample_per_epoch):
            obs_norm = obs_normalizer.transform(torch.from_numpy(obs).float().to(args.device)).cpu().numpy()
            action, logprob = get_action(obs_norm)
            next_obs, reward, done, _ = env.step(action)
            # traj_return += reward
            traj_length += 1
            tot_env_step += 1
            
            value = get_value(obs_norm)
            
            sample_ph["obs"][env_step] = obs
            sample_ph["action"][env_step] = action
            sample_ph["logprob"][env_step] = logprob
            sample_ph["next_obs"][env_step] = next_obs
            sample_ph["reward"][env_step] = reward
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
        
        if i_epoch < args.warmup_epoch:
            obs_torch = torch.from_numpy(sample_ph["obs"]).float().to(args.device)
            obs_normalizer.update(obs_torch)
            continue
        # sample_ph["obs"] = obs_normalizer.transform(obs_torch).cpu().numpy()
        buffer.add_samples(sample_ph)
        data_batch = buffer.random_batch(0)
        data_batch["obs"] = obs_normalizer.transform(torch.from_numpy(data_batch["obs"]).float().to(args.device)).cpu().numpy()
        train_loss = update(data_batch)

        if i_epoch % args.eval_interval == 0:
            traj_lengths = []
            traj_returns = []
            for traj_id in range(args.eval_num_traj):
                traj_return = traj_length = 0
                state, done = env.reset(), False
                for step in range(args.max_traj_length):
                    state_norm = obs_normalizer.transform(torch.from_numpy(state).float().to(args.device)).cpu().numpy()
                    action, _= get_action(state_norm, deterministic=True)
                    state, reward, done, _ = env.step(action)
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
            
        obs_torch = torch.from_numpy(sample_ph["obs"]).float().to(args.device)
        obs_normalizer.update(obs_torch)  

5. Record the results
---------------------

The actor's update function will return with a dict recording several metrics of the training process, and we can just \
log the statistics with one line of code:

.. code-block:: python

    logger.log_scalars("", train_loss, step=i_epoch)

Here ``i_epoch`` is the count of training epochs, ``""`` means we identifies the statistics with keys of ``train_loss``. After training is done, \
you can check the curves by typing 

.. code-block:: bash

    tensorboard --logdir </path/to/log> --bind_all

in the terminal. 