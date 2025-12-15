"""Utility functions for PSA attention."""

import torch
from functools import wraps


def timeit(func):
    """Decorator to measure execution time."""
    import time
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        ret = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        print(f"{func.__name__} execution took {(end - start)*1000:.4f}ms")
        return ret
    return wrapper


def preserve_rng_state(func):
    """Decorator to preserve random state across function call."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save current random state
        cpu_state = torch.get_rng_state()
        cuda_states = []
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                cuda_states.append(torch.cuda.get_rng_state(device))
        try:
            # Execute the decorated function
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore random state
            torch.set_rng_state(cpu_state)
            if torch.cuda.is_available():
                for device, state in enumerate(cuda_states):
                    torch.cuda.set_rng_state(state, device)
    return wrapper
