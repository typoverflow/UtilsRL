import functools

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