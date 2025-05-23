import datetime
from time import gmtime, sleep, strftime
import time
import click
import os
import signal
import subprocess
import gpustat
import fcntl
import sys
import threading

start = time.time()
DEFAULT_MUTEX_PATH = "/tmp/laizj-ysqd-mutex-{}.lock"
mutexs = []


class MutexLock:
    def __init__(self, filename=DEFAULT_MUTEX_PATH):
        self.filename = filename
        print("creating lock", filename, os.path.exists(filename))
        self.handle = open(filename, 'w')
        self.acquired = 0
        # 设置 FD_CLOEXEC 标志
        # flags = fcntl.fcntl(self.handle, fcntl.F_GETFD)
        # fcntl.fcntl(self.handle, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)

    def acquire(self):
        """Acquire the lock."""
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.acquired = 1
            return True
        except BlockingIOError:
            if self.acquired:
                self.acquired += 1
                return True
            else:
                return False
        except Exception as e:
            print(f"检测文件锁时发生错误: {e}")
            exit(-1)

    def release(self):
        """Release the lock."""
        self.acquired -= 1
        if self.acquired == 0:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)

    # def __del__(self):
    #     try:
    #         self.handle.close()
    #         os.remove(self.filename)
    #     except Exception as e:
    #         pass

def init_mutex():
    gpu_stats = gpustat.new_query()
    for i in range(len(gpu_stats)):
        mutexs.append(MutexLock(DEFAULT_MUTEX_PATH.format(i)))

init_mutex()
print()
for i in range(len(mutexs)):
    print(mutexs[i].acquire())