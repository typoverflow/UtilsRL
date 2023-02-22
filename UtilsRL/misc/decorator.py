import functools
import time
import atexit
import torch
from typing import Callable, Any, Optional


def depreciated(func):
    """
    Decorator to print depreciated message.
    """
    def wrapper(*args, **kwargs):
        print("Depreciated function: {}".format(func.__name__))
        return func(*args, **kwargs)

    return wrapper


def untested(func):
    """
    Decorator to hint that the function is untested.
    """
    def wrapper(*args, **kwargs):
        print("Untested function: {}".format(func.__name__))
        return func(*args, **kwargs)

    return wrapper


def retry(retry_times, fallback):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try_times = 0
            while try_times < retry_times:
                try_times += 1
                try:
                    ret_obj = func(*args, **kwargs)
                    return ret_obj
                except KeyboardInterrupt as k:
                    raise k
                except Exception as e:
                    print(
                        f"Exception occurred in {func.__name__}, has tried {try_times} times"
                    )
                    continue
            print(
                f"Exception occurred in {func.__name__}, calling fallback {fallback.__name__}"
            )
            if fallback is not None:
                return fallback(*args, **kwargs)
        return wrapper
    return decorator


def fallback(func):
    return retry(1, func)


class profile(object):
    """
    Decorator to profile the function and to print elapsed time at the exit of the program
    """
    global_registry = {}

    def __init__(self, name: Optional[str]=None, activate: bool=True) -> None:
        self.name = name
        self.activate = activate

        def exit_logger():
            from UtilsRL.logger import logger  # the input is moved here to avoid circular import
            torch.cuda.synchronize()
            logger.info(
                f"[profile]: Executed {self.name} {self.called_times} times, total elapsed time {self.total_time}(s), avg {self.total_time/self.called_times if self.called_times else 0}(s)."
            )

        if self.activate:
            atexit.register(exit_logger)

    def __call__(self, func: Callable) -> Callable:
        if self.activate:
            if self.name is None:
                self.name = func.__name__
            profile.global_registry[self.name] = [0, 0]

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch.cuda.synchronize()
                beg = time.time()
                ret = func(*args, **kwargs)
                torch.cuda.synchronize()
                profile.global_registry[self.name][0] += time.time() - beg
                profile.global_registry[self.name][1] += 1
                return ret

            return wrapper
        else:
            return lambda fn: fn

    def __enter__(self) -> None:
        if self.activate:
            if self.name is None:
                self.name = "default"
            torch.cuda.synchronize()
            self.beg = time.time()

    def __exit__(self, exc_type, exc_value: Any, traceback: Any) -> None:
        if self.activate:
            torch.cuda.synchronize()
            profile.global_registry[self.name][0] += time.time() - self.beg
            profile.global_registry[self.name][1] += 1
        
