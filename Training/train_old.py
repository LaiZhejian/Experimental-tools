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
        self.handle = open(filename, 'w')
        self.acquired = False
        # 设置 FD_CLOEXEC 标志
        flags = fcntl.fcntl(self.handle, fcntl.F_GETFD)
        fcntl.fcntl(self.handle, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)

    def acquire(self):
        """Acquire the lock."""
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.acquired = True
            return True
        except BlockingIOError:
            return self.acquired
        except Exception as e:
            print(f"检测文件锁时发生错误: {e}")
            exit(-1)
            

    def release(self):
        """Release the lock."""
        self.acquired = False
        fcntl.flock(self.handle, fcntl.LOCK_UN)

    def __del__(self):
        try:
            self.handle.close()
            os.remove(self.filename)
        except Exception as e:
            pass

def init_mutex():
    gpu_stats = gpustat.new_query()
    for i in range(len(gpu_stats)):
        mutexs.append(MutexLock(DEFAULT_MUTEX_PATH.format(i)))


def get_available_devices():
    device_ids = []
    gpu_stats = gpustat.new_query()
    for idx, gpu in enumerate(gpu_stats):
        if (len(gpu.processes) == 0 or gpu.memory_used < 500 or gpu.memory_available > 40000) and mutexs[idx].acquire():
            device_ids.append(idx)
    return device_ids


def set_visible_gpus(gpus: int = 1, wait_until_available: bool = False) -> None:
    global start
    while True:
        device_ids = get_available_devices()
        if len(device_ids) >= gpus:
            if wait_until_available:
                print(f'waiting for gpu {strftime("%H:%M:%S", gmtime(time.time() - start))} ...')
            break
        else:
            if wait_until_available:
                sleep(1)
                print(
                    f'waiting for gpu {strftime("%H:%M:%S", gmtime(time.time() - start))} ...\r', end='')
            else:
                print(
                    f'No enough gpu! {gpus} is required, but only {len(device_ids)} is left')
                return None
    device_ids = device_ids[:gpus]
    if not wait_until_available:
        print(f'Get device id {device_ids} at time {strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
    return os.environ['CUDA_VISIBLE_DEVICES']


@click.command()
@click.option('--bash-path', required=True, type=str)
@click.option('--gpus', default=1, type=int)
@click.option('--now', default=False, is_flag=True, type=bool)
@click.argument('args', nargs=-1)  # 捕获所有额外的参数
def main(bash_path, gpus, now, args):
    global start
    init_mutex()
    start = time.time()
    check_for_others = False
    while True:
        avai_cuda_devices = set_visible_gpus(gpus=gpus, wait_until_available = not check_for_others)
        if check_for_others:
            if avai_cuda_devices:
                break
            print('Failed to get gpu, keep waiting ...')
            check_for_others = False
        else:
            check_for_others = True
            if not now:
                print(
                    f'waiting for checking device id {avai_cuda_devices} at time {strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
                sleep(300)

    print()
    print('#' * 30, "\033[92mSUCCESS\033[0m", '#' * 30)
    print(f'Script path: {bash_path}')

    # 打印参数内容
    if args:
        print('Arguments:')
        for i, arg in enumerate(args, start=1):
            print(f'  Arg {i}: {arg}')
    else:
        print('No additional arguments provided.')

    # 打印 GPU 数量和 ID
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    gpu_count = len(gpu_ids.split(',')) if gpu_ids else 0
    print(f'GPU count: {gpu_count}, GPU IDs: [{gpu_ids}]')

    print('#' * 69)
    print()

    if os.path.splitext(bash_path)[-1] == '.py':
        command = [sys.executable, bash_path] + list(args)
        os.execve(sys.executable, command, os.environ)
    else:
        command = ['/bin/bash', bash_path] + list(args)
        os.execve('/bin/bash', command, os.environ)


if __name__ == "__main__":
    main()