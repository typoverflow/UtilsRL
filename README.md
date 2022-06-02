# UtilsRL

`UtilsRL` is a reinforcement learning utility python package, which is designed for fast integration into other RL projects. Despite its lightweightness, it still provides a full set of functions needed for RL algorithms development. 

Currently `UtilsRL` is maintained by researchers from [LAMDA-RL](https://github.com/LAMDA-RL) group. Any bug report / feature request / improvement is appreciated. We also thank [@YuRuiii](https://github.com/YuRuiii), [@cmj2020](https://github.com/cmj2002), [@paperplane03](https://github.com/paperplane03) and [@momanto](https://github.com/momanto) for their participation in code testing and performance benchmarking. 

## Installation
You can install this package directly from pypi:
```shell
pip install UtilsRL
```
After installation, you may still need to configure some other dependencies based on your platform, such as PyTorch.

## Features & Usage
### 1. RL Modules
See sub-modules in `UtilsRL/rl` for details. We provide
+ **Actors** with different output head, such as gaussian distribution, categorical distributions. 
+ **Critics**. For now we only implement single critic, but we have plans for  implementing ensemble critics. 
+ **Buffers**, which replay transitions or trjectories. 
+ **Normalizers**, which normalizes data in multiple modes. 


### 2. Experiment Managing
We provide a handful of functions for experiment management. All of them are placed under `UtilsRL.exp`, including: 
#### 2.1 Argument Parsing
The argument parsing utils in this package provides three features:
1. **Supporting for multiple types of config files.** `parse_args` can parse json, yaml, or even a python config module which is imported ahead
   ```python
   from UtilsRL.exp import parse_args
   json_config = "/path/to/json"
   yaml_config = "/path/to/yaml"
   import config_module

   json_args = parse_args(json_config)
   yaml_Args = parse_args(yaml_config)
   module_args = parse_args(config_module)
   also_module_args = parse_args("/path/to/module")
   ```
2. **Nested argument parsing.** We do this by introducing the `NameSpace` class. To be specific, if you pass `convert=True` to `parse_args`, then all of the dicts in the config file (including the argument dict itself) will be converted to a subclass of `NameSpace`. The contents wrapped in `NameSpace` can be accessed both in *dict* manner and in *attribute* manner, and they will be formatted for better illustration when printing. For example, if we define a config module as follows:
    ```python
    # in config module: config_module
    from UtilsRL.misc.namespace import NameSpace

    batch_size = 256
    num_epochs = 10
    class TrainerArgs(NameSpace):
        learning_rate = 1e-3
        weight_decay = 1e-5
        momentum = 0.9
        
    class ActorArgs(NameSpace):
        epsilon = 0.05
        class NetArgs(NameSpace):
            layer_num = 2
            layer_nodes = 256
    ```
    The we import and parse it in `main.py`:
    ```python
    import config_module
    args = parse_args(config_module)
    print(args)
    print(">>>>>>>>>>>>>>>>>>>>>>")
    print(args.TrainerArgs.learning_rate)
    ```
    The outputs are
    ```python
    <NameSpace: args>
    |ActorArgs:     <NameSpace: ActorArgs>
                    |epsilon: 0.05
                    |NetArgs:       <NameSpace: NetArgs>
                                    |layer_num: 2
                                    |layer_nodes: 256
    |TrainerArgs:   <NameSpace: TrainerArgs>
                    |learning_rate: 0.001
                    |weight_decay: 1e-05
                    |momentum: 0.9
    |batch_size: 256
    |num_epochs: 10
    >>>>>>>>>>>>>>>>>>>>>>
    0.001
    ```
3. **Argument updating.** When calling `UtilsRL.exp.parse_args`, the command-line argument is parsed and used to overwrite the arguments defined in config file. If the specific argument is nested, then you can use dot `.` to separate each NameSpace, like `python main.py --TrainerArgs.momentum 0.8`. 
    ```python
    from UtilsRL.exp import parse_args

    # get arguments from file/config module
    # this will automatically parse arguments from command line and update them to args
    args = parse_args("/path/to/file")
    
    # manually update with command line arguments
    args = update_args(args, unknown)
    ``` 

#### 2.2 Device and Seed Management
We provide a set of utils functions of selecting device and setting seed in `UtilsRL.exp.device` `UtilsRL.exp.seed`. Please take time and check these files. 

A *setup* function is available in top-level `UtilsRL.exp`, which will setup the arguments with logger, device and seed which you provide. 
```python
from UtilsRL.exp import setup

setup(args, logger=None, device="cuda:0", seed=None)  # seed will be initialized randomly
setup(args, logger=None, device=None, seed="4234")  # a most free gpu will be selected as device
```

#### 2.3 Snapshot
You can make a snapshot of the experiment code by passing `--UtilsRL.snapshot <name>` to the program. UtilsRL will commit all the changes to a new branch whose name is `<name>`, and then return to the original branch. 
After creating the branch, its name will be added to `args`. You can find its name by `args.UtilsRL.snapshot_branch`, and git diff that branch later to checkout the changes from which you made. 


### 3. Monitor
Monitor listens at the main loop of the training process, and displays the process with tqdm meter. 
```python
from UtilsRL.monitor import Monitor

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
    local_var = monitor.register_context("local_var", save_every=100)
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

### 4. Logger
Logger provides a rather shallow capsulation for `torch.utils.tensorboard.SummaryWriter`. 

```python
from UtilsRL.logger import TensorboardLogger

# create a logger, with terminal output enabled and file logging disabled
logger = TensorboardLogger(log_dir="./logs", name="debug", terminal=True, txt=False) 

# log a sentence in color blue.
logger.log_str("This is a sentence", type="LOG")
# log sentence in color red. 
logger.log_str("Here occurs an error", type="ERROR") 

# log scalar and a dict of scalars repectively
logger.log_scalar(tag="var_name", value=1.0, step=1)
logger.log_scalars(main_tag="group_name", tag_scalar_dict={
    "var1": 1.0, 
    "var2": 2.0
}, step=1)
```

