from .base_logger import BaseLogger
from .text_logger import ColoredLogger, FileLogger
from .tensorboard_logger import TensorboardLogger
from .wandb_logger import WandbLogger
from .composite_logger import CompositeLogger

logger = ColoredLogger() # this is the default logger for URL internal usage