UtilsRL.exp
===========

Experiment managing, from our perspective, basically amounts to two things: 1. `Argument Parsing` and 2. `Experiment Setup`. 

Argument Parsing
----------------

In UtilsRL, we recommend the users to use a combination of configuration file and command-line argument to prepare the experiment's parameters. \
A high-level function, :func:`~UtilsRL.exp.parse_args` takes care of this. 

.. UtilsRL.exp.parse_args
.. ~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: UtilsRL.exp.parse_args

Basicly :func:`~UtilsRL.exp.parse_args` parses the config file and use the command-line argument (for example ``--learning_rate 0.003``) to overwrite. 

But What is a :class:`~UtilsRL.misc.NameSpace`? :class:`~UtilsRL.misc.NameSpace` is a meta class we introduce to make life easier. It somewhat resembles ``Enum`` class, which allow users to define nested arguments and \
access them easily. All you need to do is wrap a group of arguments within a subclass of :class:`~UtilsRL.misc.NameSpace`

.. code-block:: python

    # suppose this is a configuration file named `args.py`
    from UtilsRL.misc import NameSpace
    class ActorArgs(NameSpace):
        num_layers = 2
        num_dim = 256
        class TrainerArgs(NameSpace):
            lr = 3e-4

Here we defined a `nested argument`, where `TrainerArgs`, `num_layers` and `num_dim` are wrapped inside `ActorArgs`. The point of doing so is basically two folds:

- | Just ``args = parse_args(); print(args.ActorArgs)``, we get this. Nice and pretty. 

::

    <NameSpace: ActorArgs>
    |num_layers: 2
    |num_dim: 256
    |TrainerArgs:   <NameSpace: TrainerArgs>
    |               |lr: 0.0003


- Both ``args.ActorArgs.num_layers`` and ``args["ActorArgs"]["num_layers"]`` are supported when extracting the arguments. 

.. note::

    Nested mode is also supported in command-line argument. For example, pass ``--ActorArgs.num_dim 3`` will overwrite the default value to 3.


Experiment Setup
----------------

Such much for parsing, now we move on to experiment setup. We mainly focus on the setup of device, seed and logger for RL experiments, and \
provide a powerful function :func:`~UtilsRL.exp.setup` as well. 

.. autofunction:: UtilsRL.exp.setup

After calling :func:`~UtilsRL.exp.setup`, you can access the selected device / seed / logger with ``args.xxxx``. We also allow the users to access these \
arguments as global variables

.. code-block:: python

    import UtilsRL.exp as exp
    args = setup(args, _logger, _device, _seed)

    args.device, args.seed, args.logger
    exp.device, exp.seed, exp.logger  # equal to the above


Some other features?
--------------------

We allow users to add features concerning experiment managing by registering custom functions. This is done by :func:`~UtilsRL.exp.argparse.register_argparse_callback`:

.. autofunction:: UtilsRL.exp.argparse.register_argparse_callback

We provide a snapshot feature as an example. This feature is enable whenever ``args`` contains the key ``UtilsRL.snapshot``, and the corresponding value will be passed \
as argument of the callback function :func:`~UtilsRL.exp.snapshot.make_snapshot`. 

.. autofunction:: UtilsRL.exp.snapshot.make_snapshot


.. note:: 
    
    Don't forget to add a line of code ``register_argparse_callback("some_key", some_callback)`` at the end of ``UtilsRL/exp/__init__.py`` when you are adding features !



Snapshot
~~~~~~~~

You can make a snapshot of the code by passing ``--UtilsRL.snapshot <name>`` to the program. \
UtilsRL will commit all the changes to a new branch whose name is ``<name>``, and then return to the original branch. After creating the branch, its name will be added to ``args``. \
You can find its name by ``args.UtilsRL.snapshot_branch``, and git diff that branch later to checkout the changes you made.

Custom Float Precision
~~~~~~~~~~~~~~~~~~~~~~

You can change the default float precision of torch by passing ``--UtilsRL.ftype <ftype>`` to the program to set custom float precision. \
Valid values include: ``float16``, ``float32``, ``float``, ``float64`` and ``double``. This callback function will set torch default float precision and \
populate the arguments. There is no way, however, to change the default precision of ``numpy`` from ``double`` to ``float``, and it requires manual \
handling for ``numpy``. 