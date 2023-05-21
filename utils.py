__author__ = "Biswa Sengupta"

import yaml
from functools import wraps
import time


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper
