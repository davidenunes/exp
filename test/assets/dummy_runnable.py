import time
import random


def run(x=1, **kwargs):
    try:
        time.sleep(random.uniform(0, 2))
        return x ** 2
    except:
        pass
