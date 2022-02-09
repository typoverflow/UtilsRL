import sys
from UtilsRL.third_party.tqdm import tqdm_tty, tqdm_notebook, tqdm_file
from UtilsRL.logger import BaseLogger
from typing import Optional, Sequence, Union

tqdm_cls = None
try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        tqdm_cls = tqdm_notebook
    if 'terminal' in ipy_str:
        tqdm_cls = tqdm_tty
except:
    if sys.stderr.isatty():
        tqdm_cls = tqdm_tty
    else:
        tqdm_cls = tqdm_file


class MonitorError(Exception):
    pass

class Monitor(object):
    def __init__(self,
                 desc: Optional[str] = None, 
                 logger: BaseLogger = None, 
                 *args, **kwargs):
        """Monitor the training iteration. 

        Args: 
            desc: Description of the Monitor.
            logger: Hook of the Logger object for internal use.
        
        """
        self.desc = desc if tqdm_cls == tqdm_file else "\033[1;37m[{}]\033[0m".format(desc)
        self.tqdm_cls = tqdm_cls
        self.logger = logger
        self.args = args
        self.kwargs = kwargs

    def listen(self,
                iterable = None, 
                initial: Optional[int] = 0, 
                load: Union[bool, str] = False, 
                total: Optional[int] = None, 
                miniters: Optional[int] = None):
        """Set the monitor to listen at a certain iteration. Note that
           a monitor can be assigned to listening only once. 
        """

        if hasattr(self, "iterable") and self.iterable is not None:
            raise MonitorError("A monitor can only listen at one iterable.")
        
        self.iterable = iterable
        self.tqdm = self.tqdm_cls(iterable, self.desc, total=total, initial=initial, miniters=miniters)
        self.total = self.tqdm.total
        self.initial = initial
        self.miniters = self.tqdm.miniters
        
        self.global_step = self.initial
        return self

    def __iter__(self):
        """Manully iterate self.tqdm. Note that we don't return self.tqdm
            in self.listem so as to trace the global step.  
        """
        for obj in self.tqdm:
            yield obj
            self.global_step += 1
