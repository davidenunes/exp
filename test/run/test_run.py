import sys
import os
import random


def run(param1=None, param2=None, **kargs):
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]

    # TODO test exceptions
    if random.random() < 0.3:
        raise Exception("failled with params: \n{}".format(kargs))

    return "INSIDE GPU WORKER ---------------\n" \
           "params: \n " \
           "misc: {kargs}\n" \
           "{p1} \n" \
           "{p2} \n" \
           "using GPU: {env}\n".format(p1=param1, p2=param2, env=gpu, kargs=kargs)


if __name__ == "__main__":
    # note I can use argparse in the scripts to run directly from main
    run()
