import time
from tqdm import tqdm

progress = tqdm(total=20)



for i in range(20):
    time.sleep(0.5)

    progress.update(n=1)

    #print(progress.avg_time)

    # print(progress.start_t)
    # print(progress.)
    print((progress.total - progress.n) * progress.avg_time)

progress.close()