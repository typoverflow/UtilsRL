# UtilsRL

`UtilsRL` is a reinforcement learning utility python package, which is designed for fast integration into other RL projects. Despite its lightweightness, it still provides a full set of functions needed for RL algorithms development. 

Currently `UtilsRL` is maintained by researchers from [LAMDA-RL](https://github.com/LAMDA-RL) group. Any bug report / feature request / improvement is appreciated.

## Installation
You can install this package directly from pypi:
```shell
pip install UtilsRL
```
After installation, you may still need to configure some other dependencies based on your platform, such as PyTorch.

## Features & Usage
<!-- See [the documentation](https://utilsrl.readthedocs.io) for details.  -->
We are still working on the docs, and the docs will be published as soon as possible.  

Here we list some highlight features of UtilsRL:
- **Extremely easy-to-use and research friendly argument parsing**. `UtilsRL.exp.argparse` supports several handy features for research:
  - loading arguments from both `yaml`, `json`, `python` files and command line
  - nested argument parsing
- **Well-implemented torch modules for Reinforcement Learning**
  - common network structures: MLP, CNN, RNN, Attention, Ensemble Blocks and etc
  - policy networks with various output distributions
  - normalizers implemented in `nn.Module`, benefiting saving/loading by taking advantage of `state_dict`
- **Powerful experiment loggers**.
- **Super fast Prioritized Experience Replay (PER) buffer**. By binding c++-implemented data structures, we boost the efficiency of PER up to 10 times

We provide two examples, namely training PPO on mujoco tasks and training Rainbow on atari tasks as illustrations for integrating UtilsRL into your workflow (see `examples/`)

## Acknowledgements
We took inspiration for module design from [tianshou](https://github.com/thu-ml/tianshou) and [Polixir OfflineRL](https://github.com/polixir/OfflineRL).

We also thank [@YuRuiii](https://github.com/YuRuiii) and [@momanto](https://github.com/momanto) for their participation in code testing and performance benchmarking. 
