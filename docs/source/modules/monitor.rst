UtilsRL.monitor
===============

Monitor is designed to like a manager to monitor the progress of the experiment. For basic usage, it can be used as ``tqdm.tqdm``, which \
wraps an iterable object and prints a progress bar on the screen. We enhanced ``Monitor`` with some more features on top of this, namely

* **Context Variable Registration.** Sometimes we hope to save products (like model checkpoint) or temporary states (like replay buffer) for reproducibility and staged training, and then ``Monitor`` may provide some help. You can register a `context variable` with :func:`~UtilsRL.monitor.Monitor.register_context`, then the context variables will be saved at an interval of ``save_every``. You can also resume training with such method in the following snippet, where we defined a class ``Trainer``, load ``actor`` and ``local_var`` from path `/path/to/ck_dir` and resume training from the ``initial``-th iteration. 

.. code-block:: python

        class Trainer():
            def __init__(self):
                self.actor = ...
            
            def train(self):
                local_var = ...
                
                # load previous saved checkpoints specified by `load_path`
                self.actor, local_var = \
                    monitor.register_context(["self.actor", "local_var"], load_path="/path/to/ck_dir").values()
                # use `initial` to designate the start point
                for i_epoch in monitor.listen(range(1000), initial=100):
                    # continue training


* **Callback Function Registration.** You can register callback functions which will be triggered at certain state of the training with :func:`~UtilsRL.monitor.Monitor.register_callback`. For example, a function which will email us when training is completed. 

.. code-block:: python

        monitor = Monitor(desc="test_monitor")
        monitor.register_callback(
            name= "email me at the end of training", 
            on = "exit", 
            callback = Monitor.email, 
            ...
        )


UtilsRL.monitor.Monitor
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: UtilsRL.monitor.Monitor
    :members:
    :show-inheritance: