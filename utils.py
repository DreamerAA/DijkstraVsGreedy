import time
from typing import Callable


def st_time(func: Callable[..., None], run_count:int = 1) -> Callable[..., None]:
    """
    st decorator to calculate the total time of a func
    """

    def st_func(*args: str, **keyArgs: int) -> None:
        t1 = time.time()
        for _ in range(run_count):
            func(*args, **keyArgs)
        t2 = time.time()
        full_time = t2 - t1
        print(
            f"Function = {func.__name__}: Time one run = {full_time/run_count}"
        )

    return st_func
