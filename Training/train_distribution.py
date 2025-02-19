import datetime
from time import gmtime, sleep, strftime
import time
import click
import os
import subprocess
import gpustat
import fcntl

start = time.time()
DEFAULT_MUTEX_PATH = "/tmp/laizj-ysqd-mutex.lock"

hosts = [20080,
         20687,
         20227,
         20638,
         20658,
         20659,
         20641,
         20644,
         20642,
         20645,
         20646,
         20647,
         21469,
         21472]



class MutexLock:
    def __init__(self, filename=DEFAULT_MUTEX_PATH):
        self.filename = filename
        self.handle = open(filename, 'w')

        # 设置 FD_CLOEXEC 标志
        flags = fcntl.fcntl(self.handle, fcntl.F_GETFD)
        fcntl.fcntl(self.handle, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)

    def acquire(self):
        """Acquire the lock."""
        fcntl.flock(self.handle, fcntl.LOCK_EX)

    def release(self):
        """Release the lock."""
        fcntl.flock(self.handle, fcntl.LOCK_UN)

    def __del__(self):
        try:
            self.handle.close()
            os.remove(self.filename)
        except Exception as e:
            pass


@click.command()
@click.option('--bash-path', required=True, type=str)
@click.option('--gpus', default=1, type=int)
@click.option('--retry', default=False, is_flag=True, type=bool)
def main(bash_path, gpus, retry):
    global start
    start = time.time()

    mutex = MutexLock()
    mutex.acquire()
    
    while True:
        
        for host in hosts:
            return_value = os.system(
                f"ssh -i exp_server -p {host} 210.28.133.13 '/home/nfs01/anaconda3/envs/laizj-torch1.10-py38/bin/python /home/nfs03/laizj/code/util/Training/distribution.py --bash-path {bash_path} --gpus {gpus}'")
            if return_value:
                print(f"host {host} failed")
                print(f'waiting for gpu {strftime("%H:%M:%S", gmtime(time.time() - start))} ...')
                sleep(5)
            else:
                print(f"Using the gpus on host:{host}")
                sleep(40)
                mutex.release()
                print("\nreleased lock")
                exit(0)
        if not retry:
            break

if __name__ == "__main__":
    main()
