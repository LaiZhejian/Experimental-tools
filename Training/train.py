import datetime
import socket
from time import gmtime, sleep, strftime
import time
from typing import List
import click
import os
import gpustat
import fcntl
import sys
import subprocess
from .send_email import send_experiment_completion_email

start = time.time()
DEFAULT_MUTEX_PATH = "/tmp/laizj-ysqd-mutex-{}.lock"
mutexs = []


class MutexLock:
    def __init__(self, filename=DEFAULT_MUTEX_PATH):
        self.filename = filename
        # print("creating lock", filename, os.path.exists(filename))
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

    def __del__(self):
        try:
            self.handle.close()
            # os.remove(self.filename)
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
        if (len(gpu.processes) == 0 or gpu.memory_available / gpu.memory_total > 0.9) and gpu.memory_used < 5000 and mutexs[idx].acquire():
            device_ids.append(idx)
    return device_ids


def set_visible_gpus(gpus=1, no_wait=False): # 一直寻找，直到找到指定数量的GPU，并对这些GPU上锁
    global start
    device_ids = None
    while device_ids is None:
        print(f'waiting for gpu {strftime("%H:%M:%S", gmtime(time.time() - start))} ...\r', end='')
        if isinstance(gpus, int):
            device_ids = get_available_devices()
            if len(device_ids) >= gpus:
                for ids in device_ids[gpus:]:
                    mutexs[ids].release()
                device_ids = device_ids[:gpus]
            else:
                for ids in device_ids:
                    mutexs[ids].release()
                device_ids = None
        else:
            flag = True
            for ids in gpus:
                if mutexs[ids].acquire():
                    mutexs[ids].release()
                else:
                    flag = False
                    break
            if flag:
                for ids in gpus:
                    mutexs[ids].acquire()
                device_ids = gpus
        if device_ids is None and no_wait:
            print('\nno gpu available, exit')
            exit(-1)
    print(f'waiting for gpu {strftime("%H:%M:%S", gmtime(time.time() - start))} ...')
    return device_ids


@click.command()
@click.option('--bash-path', required=True, type=str)
@click.option('--gpus', default=1, type=int)
@click.option('--now', default=False, is_flag=True, type=bool)
@click.option("--no_wait", default=False, is_flag=True, type=bool)
@click.option('--email', default=False, is_flag=True, type=bool)
@click.argument('args', nargs=-1)  # 捕获所有额外的参数
def main(bash_path, gpus, now, no_wait, email, args):
    global start
    
    if len(gpustat.new_query()) == 0:
        print('#' * 30, "\033[92mFAILURE\033[0m", '#' * 30)
        exit(0)
    
    init_mutex()
    start = time.time()
    check_for_others = False
    while True:
        avai_cuda_devices = set_visible_gpus(gpus=gpus if isinstance(gpus, int) else len(gpus), no_wait=no_wait)
        if now:
            gpus = avai_cuda_devices
            break
        else:
            if check_for_others:
                if avai_cuda_devices == gpus:
                    break
                else:
                    for ids in gpus:
                        mutexs[ids].release()
                    gpus = avai_cuda_devices
                    print('Failed to get gpu, keep waiting ...')
            else:
                gpus = avai_cuda_devices
                check_for_others = True
                print(f'Waiting for checking device id {avai_cuda_devices} at time {strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
                sleep(300)
                
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

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

    start = time.time()
    if os.path.splitext(bash_path)[-1] == '.py':
        command = [sys.executable, bash_path] + list(args)
        # os.execve(sys.executable, command, os.environ)
    else:
        command = ['/bin/bash', bash_path] + list(args)
    
    run_exception = None
    try:
        subprocess.run(command, env=os.environ, check=True)
    except Exception as e:
        run_exception = e

    if email:
        arg_content = [f"Arg {i}: {arg}" for i, arg in enumerate(args, start=1)]
        arg_content = " ".join(arg_content) if arg_content else "no argument"
        
            # ======== 邮件内容 ========
        hostname = socket.gethostname()
        completion_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if run_exception:
            subject_info = "[实验失败]"
            email_content = f"""
        您的实验出现了错误！
        
        实验名称: {os.path.basename(bash_path)}
        实验参数: {arg_content}
        错误信息: {run_exception}
        完成时间: {completion_time}
        运行主机: {hostname} GPUS: {gpus}
        
        """
        else:
            subject_info = "[实验完成]"
            
            elapsed_time = int(time.time() - start)
            hours = elapsed_time // 3600
            minutes = (elapsed_time % 3600) // 60
            seconds = elapsed_time % 60
            
            email_content = f"""
        您的实验已顺利完成！
        
        实验名称: {os.path.basename(bash_path)}
        实验参数: {arg_content}
        运行耗时: {hours:02d}:{minutes:02d}:{seconds:02d}
        完成时间: {completion_time}
        运行主机: {hostname} GPUs: {gpus}

        此邮件由实验监控脚本自动发送，请勿直接回复。
        """
        
        send_experiment_completion_email(
            source_name="实验监控系统",
            subject=f"{subject_info} - {os.path.basename(bash_path)}",
            email_content=email_content
        )
        
    # if os.path.splitext(bash_path)[-1] == '.py':
    #     command = [sys.executable, bash_path] + list(args)
    # else:
    #     command = ['/bin/bash', bash_path] + list(args)
    # subprocess.run(command, env=os.environ, check=True)

if __name__ == "__main__":
    main()