import functools
import time
import atexit

from UtilsRL.logger import logger

def depreciated(func):
    """
    Decorator to print depreciated message
    """
    def wrapper(*args, **kwargs):
        print("Depreciated function: {}".format(func.__name__))
        return func(*args, **kwargs)
    return wrapper

def untested(func):
    """_summary_
    Decorator to hint that the function is untested
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
                    print(f"Exception occurred in {func.__name__}, has tried {try_times} times")
                    continue
            print(f"Exception occurred in {func.__name__}, calling fallback {fallback.__name__}")
            if fallback is not None:
                return fallback(*args, **kwargs)
        return wrapper
    return decorator

def fallback(func):
    return retry(1, func)

    
def profile(func):
    called_times = 0
    elapsed_time = 0
    def exit_logger():
        logger.log_str(f"[Profile]: Function {func.__name__}, called {called_times} times, total elapsed time {elapsed_time}(s), avg {elapsed_time/called_times if called_times else 0}(s).", type="WARNING")
    atexit.register(exit_logger)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        beg = time.time()
        ret = func(*args, **kwargs)
        nonlocal called_times
        nonlocal elapsed_time
        elapsed_time += time.time() - beg
        called_times += 1
        return ret
    return wrapper