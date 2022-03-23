import os
import sys
import copy
import smtplib
import contextlib
import atexit
import inspect
import pickle
from smtplib import SMTP_SSL, SMTP
from email.mime.text import MIMEText

from UtilsRL.third_party.tqdm import tqdm_tty, tqdm_notebook, tqdm_file
from UtilsRL.logger import BaseLogger, DummyLogger

from typing import Optional, Sequence, Union, Callable, Any

tqdm_cls: Any = None
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


@contextlib.contextmanager
def update_scope(func: Callable, globals: dict):
    old_globals = func.__globals__.copy()
    func.__globals__.update(globals)
    yield func
    func.__globals__.update(old_globals)
    

class MonitorError(Exception):
    pass


class Monitor(object):
    """Monitor is designed to monitor the main for loog of the training process. 
        It currently supports for 3 purposes:
        - Monitor.listen() wraps a iterable and visualize the progress meter just like tqdm does, 
            but Monitor identifies output (tty or file) and adjust its behavior accordingly.
        - Monitor.register_callback() registers a callback function which will be called when
            the condition is satisfied. A simple usage is to send yourself an email for notification 
            when training is done. 
        - Monitor.register_context() registers group of variables as `context`. Monitor will save the context 
            variables periodically and restore them if the training is resumed from a checkpoint.
    """

    @staticmethod
    def eval_outer(expr):
        L = inspect.currentframe().f_back.f_back.f_locals
        G = inspect.currentframe().f_back.f_back.f_globals
        return eval(expr, G, L)

    @staticmethod
    def email(msg, to, user, password, smtp_server=None, port=None):
        def get_host(user):
            return user.split("@")[1].split(".")[0]
        host_info = {
            "qq": ("smtp.qq.com", 587),
            # "gmail": ("smtp.gmail.com", 587), 
            "outlook": ("smtp.office365.com", 587)
        }
        if smtp_server is None or port is None:
            host = get_host(user)
            if host not in host_info:
                raise KeyError("Host {} is not supported, current supported types are: {}".format(host, list(host_info.keys())))
            smtp_server, port = host_info[host]
            
        _msg = MIMEText(msg, "plain", _charset="utf-8")
        _msg["Subject"] = "An message from your program"
        _msg["from"] = "UtilsRL.Monitor"

        with SMTP(host=smtp_server, port=port) as smtp:
            smtp.starttls()
            smtp.login(user = user, password = password)
            smtp.sendmail(from_addr=user, to_addrs=to, msg=_msg.as_string())

    def __init__(self,
                 desc: Optional[str] = None, 
                 out_dir: Optional[str] = None, 
                 logger: Optional[BaseLogger] = None, 
                 *args, **kwargs):
        """Monitor the training iteration. 

        Args: 
            desc: Description of the Monitor.
            out_dir: Output directory of the products. Must be specified if to use register_context.
            logger: Hook of the Logger object for internal use. If set to None, a dummy logger will be used.
        """
        self.desc = desc if tqdm_cls == tqdm_file else "\033[1;37m[{}]\033[0m".format(desc)
        self.tqdm_cls = tqdm_cls
        self.out_dir = out_dir
        self.logger = logger if logger else DummyLogger()
        self.args = args
        self.kwargs = kwargs

        self.has_callbacks = False
        self.callbacks = list()
        self.exit_callbacks = list()
        self.end_callbacks = list()
        self.has_context = False

    def listen(self,
                iterable = None, 
                initial: Optional[int] = 0, 
                total: Optional[int] = None, 
                miniters: Optional[int] = None):
        """Set the monitor to listen at a certain iteration. Note that a monitor can be assigned 
            for listening only once.

        Args: 
            iterable: The Iterable object to wrap up.
            initial: Startpoint of the iteration.
            total: Total number of iteration. If left None, total will be set to `len(iterable)`.
            miniters: Minimum number of iterations between two updates. If set to None, it will be set to 1.
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

    def register_callback(self, 
                          name: str, 
                          on: Optional[Union[str, int]] = None, 
                          callback: Callable = None, 
                          *args, **kwargs):
        """Register callback functions which will be called when `on` is satisfied.
        
        Args: 
            name: the name of the callback.
            on: Specifies the condition. Possibile values are:
                - None, then the callback will never be called. 
                - 'exit', then the callback will be called on exit. 
                - int, then the callback will be called at the beginning of `on`th iteration. 
                - str which represents a percentage, then the callback will be called at this
                    stage of training.  
            callback: the callback funtion. It will take args and kwargs as input, and self.global_step
                will also be added as keyward argument. So when defining a callbackj function, it's 
                better to receive redundant kwargs with `**kwargs`.
        """
        self.has_callbacks = True
        if on is None or on == False:
            return 
        elif on == 'exit':
            if name in [ec["name"] for ec in self.exit_callbacks]: 
                return
            self.exit_callbacks.append({
                "name": name, 
                "on": "exit", 
                "callback": callback, 
                "args": (args, kwargs)
            })
            atexit.register(callback, *args, **kwargs)
        elif on == "100%" or on == "end":
            if name in [ec["name"] for ec in self.end_callbacks]: 
                return
            self.end_callbacks.append({
                "name": name, 
                "on": "end", 
                "callback": callback, 
                "args": (args, kwargs)
            })
        else:
            if name in [c["name"] for c in self.callbacks]:
                return
            if isinstance(on, str):
                try:
                    per = float(on[:-1]) / 100
                    assert 0 <= per < 1
                except Exception:
                    raise MonitorError("Invalid percentage {}".format(on))
            else:
                if not isinstance(on, int):
                    raise MonitorError("Unrecognized condition: {}".format(on)) 
            self.callbacks.append({
                "name": name, 
                "on": on, 
                "callback": callback, 
                "args": (args, kwargs)
            })

    def register_context(self, expressions, save_every=None, save_mode="replace", load_path=None):
        """Register variables as context. Monitor will save the context variables 
            periodically and restore them if the training is resumed from a checkpoint.
            Note: only one register_context call with valid save_every is permitted. 

        Args:
            expressions: The expressions of the variables which you wish to designate as context. 
            save_every: save the context every `save_every` iterations. If set to None, the context 
                will not be saved. 
            save_mode: Specifies the mode of saving. Possibile values are:
                - "replace": replace previously saved context. 
                - "append": save context without replacing. 
            load_path: Specifies the path of the checkpoint of the context to load. If set to None, 
                the context will not be loaded.
        """
        
        if isinstance(expressions, str):
            expressions = [expressions]
        if save_every is None or save_every == False:
            pass
        elif isinstance(save_every, int) and save_every >= 1:
            if self.has_context:
                raise MonitorError("Only one register_context call with valid save_every is permitted.")
            if self.out_dir is None:
                raise MonitorError("Before using monitor to save context, you must specify the output directory.")
            if save_mode not in ["replace", "append"]:
                raise MonitorError("save_mode must be either 'replace' or 'append'.")
            # save the infos for context saving
            self.has_context = True
            self.context = expressions
            self.context_load_path = load_path
            self.save_every = save_every
            self.save_mode = save_mode
        else:
            raise MonitorError(f"Illegal value for save_every: {save_every}")

        ret_dict = dict()
        if load_path:
            # load obj from given path
            for expr in expressions:
                if os.path.exists(os.path.join(load_path, expr)):
                    with open(os.path.join(load_path, expr), "rb") as fp:
                        ret_dict[expr] = pickle.load(fp)
                else:
                    ret_dict[expr] = Monitor.eval_outer(expr)
        else:
            # get reference from outer scope
            for expr in expressions:
                ret_dict[expr] = Monitor.eval_outer(expr)

        return ret_dict if len(ret_dict) > 1 else ret_dict[expressions[0]]
    
    def _check_callbacks(self):
        if not hasattr(self, "global_step") or not hasattr(self, "total"):
            raise MonitorError("Monitor must listen on an iterable before registered callbacks can be called.")
        for c in self.callbacks:
            if isinstance(c["on"], int):
                if self.global_step == c["on"]:
                    c["callback"]( *c["args"][0], **c["args"][1], global_step=self.global_step)
            elif isinstance(c["on"], str): 
                threshold = int(c["on"][:-1]) / 100 * self.total
                if self.global_step >= threshold and self.global_step - 1 < threshold:
                    c["callback"](*c["args"][0], **c["args"][1], global_step=self.global_step)

    def __iter__(self):
        """Manully iterate self.tqdm. Note that we don't return self.tqdm
            in self.listem so as to trace the global step.  
        """
        tqdm_iter = iter(self.tqdm)
        while True: 
            try: 
                if self.has_context \
                    and self.global_step > self.initial \
                    and self.global_step % self.save_every == 0:
                    
                    save_path = os.path.join(self.out_dir, str(self.global_step)) if self.save_mode == "append" \
                                else os.path.join(self.out_dir, "context")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    for expr in self.context:
                        obj = Monitor.eval_outer(expr)
                        with open(os.path.join(save_path, expr), "wb") as fp:
                            pickle.dump(obj, fp)
                if self.has_callbacks: 
                    self._check_callbacks()
                yield next(tqdm_iter)
                self.global_step += 1
            except StopIteration:
                for c in self.end_callbacks:
                    c["callback"](*c["args"][0], **c["args"][1], global_step=self.global_step)
                break
