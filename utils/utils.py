import matplotlib.pyplot as plt
import time
from typing import Callable

def get_cmap(n, name:str='gist_rainbow'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


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