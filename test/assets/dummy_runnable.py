import time
import random


def run(x=1, **kwargs):
    time.sleep(random.uniform(0, 0.5))
    if x == 3:
        raise ValueError("oops!!")
    return x ** 2
