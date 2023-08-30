from .base_logger import BaseLogger
from .text_logger import FileLogger
from .tensorboard_logger import TensorboardLogger
from .wandb_logger import WandbLogger
from .composite_logger import CompositeLogger

logger = BaseLogger() # this is the default logger for URL internal usage