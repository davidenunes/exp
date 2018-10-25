import sys
import os
import tensorflow as tf
from exp.args import ParamDict,Namespace

defaults = {
    'x': (float, None),
    'id': (int, 0),
    'run': (int, 1)
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run(**kargs):
    args = ParamDict(defaults)
    args.from_dict(kargs)
    ns = args.to_namespace()
    #args = Namespace(args)
    # print(dargs)

    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    # test exceptions
    # if random.random() < 0.3:
    #   raise Exception("failled with params: \n{}".format(kargs))

    a = tf.random_uniform([100000, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    d = tf.multiply(c, ns.x)
    #d = tf.matmul(c, ns.x)

    # cfg = tf.ConfigProto(log_device_placement=True)
    # sess = tf.Session(config=cfg)
    sess = tf.Session()
    res = sess.run(d)
    sess.close()

    debug = "INSIDE GPU WORKER ---------------\n" \
            "params: {params}\n" \
            "using GPU: {env}\n " \
            "result: \n {res}" \
            "-----------------------------------".format(params=args, env=gpu, res=res)

    tf.reset_default_graph()
    return debug


if __name__ == "__main__":
    # note I can use argparse in the scripts to run directly from main
    run()
