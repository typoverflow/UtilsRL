UtilsRL.rl
==========

This sub-package contains common modules and entities in classical Reinforcement Learning pipelines. \
Ideally, we hope modules in this package can work out-of-box, so we decide to decouple their implementations and make them as lightweight as possible. \
Although currently there is only a limited set of modules, we intend to add more features in the near future, so take a look at our plans `here <https://github.com/typoverflow/UtilsRL/issues>`_!

RL Actors
---------

`Actors` are essentially the `agent` itself, they take observations as input, and output corresponding actions. We follow `tianshou`'s design of decoupling output head and \
feature extraction backend, so that we can reduce code redundancy and increase flexibility of constructing an agent. 

Basically all types of actors are composed of three parts: 

- ``backend``, which extracts hidden features from states / observations / sequences / images;
- ``output_layers``, is a :class:`UtilsRL.rl.net.MLP` object usually, whose shape is designated by ``hidden_dims``;
- `sampling strategy`, which differs from each other. 

and all actors need to implement three interfaces: 

- ``forward``, the actual forward pass of ``backend`` and ``output_layers``;
- ``sample``, sample actions from output according to the sampling strategy;
- ``evaluate``, evaluate the log-probability of actions given states. 

We list the api docs here. 

Base Actor
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UtilsRL.rl.actor.BaseActor
    :members:
    :show-inheritance:

Deterministic Actors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used in `TD3` and `DDPG`. 

.. autoclass:: UtilsRL.rl.actor.DeterministicActor
    :members:
    :show-inheritance:

.. autoclass:: UtilsRL.rl.actor.SquashedDeterministicActor
    :members:
    :show-inheritance:
    
.. autoclass:: UtilsRL.rl.actor.ClippedDeterministicActor
    :members:
    :show-inheritance:


Stochastic actors
~~~~~~~~~~~~~~~~~

Actors which will sample from a stochastic distribution, and the layers will compute the \
parameters of the distribution. For continuous control problems, `GaussianActors` are frequently used; while \
in discrete control problems, `CategoricalActors` are preferred when action space is comparatively small. 

.. autoclass:: UtilsRL.rl.actor.GaussianActor
    :members:
    :show-inheritance:

.. autoclass:: UtilsRL.rl.actor.SquashedGaussianActor
    :members:
    :show-inheritance:

.. autoclass:: UtilsRL.rl.actor.ClippedGaussianActor
    :members:
    :show-inheritance:

.. autoclass:: UtilsRL.rl.actor.CategoricalActor
    :members:
    :show-inheritance:


RL Critics
----------

For now there is only one critic in UtilsRL, which is a vanilla implementation that takes observation (or observation-action pair) as input \
and outputs value. It is composed by two consecutive modules: ``feature extraction layers -- output layers``, so as to benefit from the \
decoupling structure. 

.. autoclass:: UtilsRL.rl.critic.SingleCritic
    :members:
    :show-inheritance:


Normalizers
-----------

`Normalizers` are used to transform raw tensors into a form suitable for neural networks to process. \
In RL, we found normalizers are extremely important for on-policy algorithms and offline algorithms. 

For all normalizers we subclasses them from ``torch.nn.Module``, so that their states and parameters can be saved & loaded \
with torch utils. Also, you can use ``__call__`` to transform the data. 


.. autoclass:: UtilsRL.rl.normalizer.BaseNormalizer
    :members:
    :show-inheritance:

.. autoclass:: UtilsRL.rl.normalizer.DummyNormalizer
    :members:
    :show-inheritance:

.. autoclass:: UtilsRL.rl.normalizer.RunningNormalizer
    :members:
    :show-inheritance:

.. autoclass:: UtilsRL.rl.normalizer.StaticNormalizer
    :members:
    :show-inheritance:

.. autoclass:: UtilsRL.rl.normalizer.MinMaxNormalizer
    :members:
    :show-inheritance:



