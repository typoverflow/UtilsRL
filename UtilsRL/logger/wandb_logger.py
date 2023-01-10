# import wandb

from .base_logger import BaseLogger, make_unique_name, LogLevel

class WandbLogger(BaseLogger):
    def __init__(self, activate: bool=True, level: int=LogLevel.WARNING, *args, **kwargs):
        import wandb