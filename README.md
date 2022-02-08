# RLUtils
An python util module designed for reinforcement learning. It provides a list of util functions or classes, such as loggers, experiment monitors and etc.

## Installation
You can install this package directly from pypi:
```shell
pip install UtilsRL
```
After installation, you may still need to configure some other dependencies based on your platform, such as PyTorch.

## Usage
Logger provides a rather shallow capsulation for `torch.utils.tensorboard.SummaryWriter`. 
### Logger
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

### Monitor
Monitor is designed for monitoring the process of training and the management of certain events. 
