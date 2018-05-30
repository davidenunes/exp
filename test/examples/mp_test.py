import multiprocessing as mp
import time
import GPUtil as gpu
import random
from tqdm import tqdm

# get ids for all available gpus
gpus = [g.id for g in gpu.getGPUs()]
print("gpus: ", gpus)

tasks = range(6)
# progress_bar = tqdm(total=len(tasks))

manager = mp.Manager()
idQueue = manager.Queue()

pbar = tqdm(total=len(tasks))

# use queue to assign a gpu id to the processes
# doesn't depend on internals, each worker will work with
# one specific id
# init will share this queue
for gpu_id in gpus:
    idQueue.put(gpu_id)


def init(queue):
    global idx
    idx = queue.get()


def f(x):
    global idx

    # print("started ", idx)
    # print("input ",x)
    # process = mp.current_process()

    time.sleep(0.5 + random.random())


pool = mp.Pool(processes=1, initializer=init, initargs=(idQueue,))


def updatepbar(*res):
    """

    :param res: the results from each execution of f
    :return:
    """
    pbar.update()


for i in range(len(tasks)):
    pool.apply_async(f, args=(tasks[i],), callback=updatepbar)

pool.close()
pool.join()
pbar.close()

print("done")
