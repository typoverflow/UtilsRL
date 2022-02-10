# UtilsRL
A util python module designed for reinforcement learning. 

## Installation
You can install this package directly from pypi:
```shell
pip install UtilsRL
```
After installation, you may still need to configure some other dependencies based on your platform, such as PyTorch.

## Features & Usage
### Monitor
Monitor listens at the main loop of the training process, and displays the process with tqdm meter. 
```python
monitor = Monitor(desc="test_monitor")
for i in monitor.listen(range(5)):
    time.sleep(0.1)
```
You can **register callback functions** which will be triggered at certain stage of the training. For example, we can register a callback which will email us when training is done: 
```python
monitor = Monitor(desc="test_monitor")
monitor.register_callback(
    name= "email me at the end of training", 
    on = "exit", 
    callback = Monitor.email, 
    ...
)
```
You can also **register context variables** for training, which will be automatically managed by monitor. In the example below, the registered context variables (i.e. `self.actor` and `local_var` ) will be saved every 100 iters.
```python
monitor = Monitor(desc="test_monitor", out_dir="./out")
def train():
    local_var = ...
    local_var = monitor.register_context(local_var, save_every=100)
    for i_epoch in monitor.listen(range(1000)):
        # do training
train()
```


As a more complex example, we can use the Monitor to **resume training from a certain iteration**, and restore the context variables from checkpoints:
```python
class Trainer():
    def __init__(self):
        self.actor = ...
    
    def train(self):
        local_var = ...
        
        # load previous saved checkpoints specified by `load_path`
        self.actor, local_var = \
            monitor.register_context(["self.actor", "local_var"], load_path="/path/to/checkpoint/dir").values()
        # use `initial` to designate the start point
        for i_epoch in monitor.listen(range(1000), initial=100):
            # continue training
```

### Logger
Logger provides a rather shallow capsulation for `torch.utils.tensorboard.SummaryWriter`. 

```python
from UtilsRL.logger import BaseLogger

# create a logger, with terminal output enabled and file logging disabled
logger = BaseLogger(log_dir="./logs", name="debug", terminal=True, txt=False) 

# log a sentence in color blue.
logger.log_str("This is a sentence", type="LOG")
# log sentence in color red. 
logger.log_str("Here occurs an error", type="ERROR") 

# log scalar and a dict of scalars repectively
logger.log_scala(tag="var_name", value=1.0, step=1)
logger.log_scalas(main_tag="group_name", tag_scalar_dict={
    "var1": 1.0, 
    "var2": 2.0
}, step=1)
```

## Under Development
+ device utils
+ arg-parsing utils