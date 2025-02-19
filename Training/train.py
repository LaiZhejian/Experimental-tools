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

start = time.time()
DEFAULT_MUTEX_PATH = "/tmp/laizj-ysqd-mutex.lock"

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


def get_available_devices():
    device_ids = []
    gpu_stats = gpustat.new_query()
    for idx, gpu in enumerate(gpu_stats):
        if len(gpu.processes) == 0 or gpu.memory_used < 500 or gpu.memory_available > 40000:
            device_ids.append(idx)
    return device_ids


def set_visible_gpus(gpus: int = 1, wait_unitl_available: bool = False) -> None:
    global start
    while True:
        device_ids = get_available_devices()
        if len(device_ids) >= gpus:
            if wait_unitl_available:
                print(
                    f'waiting for gpu {strftime("%H:%M:%S", gmtime(time.time() - start))} ...')
            break
        else:
            if wait_unitl_available:
                sleep(1)
                print(
                    f'waiting for gpu {strftime("%H:%M:%S", gmtime(time.time() - start))} ...\r', end='')
            else:
                print(
                    f'No enough gpu! {gpus} is required, but only {len(device_ids)} is left')
                return None
    device_ids = device_ids[:gpus]
    if not wait_unitl_available:
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
    start = time.time()

    print("acquiring lock")
    mutex = MutexLock()
    mutex.acquire()

    # 记录子进程的 PID
    child_pid = None

    def handle_interrupt(signum, frame):
        """处理 Ctrl+C 中断信号"""
        print(f"\nInterrupt received, terminating child process {child_pid}...")
        if child_pid:
            try:
                os.kill(child_pid, signal.SIGTERM)  # 发送 SIGTERM 终止子进程
                print(f"Child process {child_pid} terminated.")
            except ProcessLookupError:
                print(f"Child process {child_pid} already terminated.")
        mutex.release()
        print("released lock")
        sys.exit(0)

    # 注册信号处理器
    signal.signal(signal.SIGINT, handle_interrupt)

    check_for_others = False
    while True:
        avai_cuda_devices = set_visible_gpus(
            gpus=gpus, wait_unitl_available=not check_for_others)
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

    pid = os.fork()
    if pid == 0:
        # 子进程逻辑
        sleep(60)
        mutex.release()
        print("\nreleased lock")
        exit(0)
    else:
        # 主进程逻辑
        child_pid = pid  # 记录子进程的 PID

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

    command = ['/bin/bash', bash_path] + list(args)
    os.execve('/bin/bash', command, os.environ)


if __name__ == "__main__":
    main()