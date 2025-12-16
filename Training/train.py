import datetime
from time import gmtime, sleep, strftime
import time
import click
import os
import subprocess
import gpustat

start = time.time()


def get_available_devices():
    device_ids = []
    gpu_stats = gpustat.new_query()
    for idx, gpu in enumerate(gpu_stats):
        if len(gpu.processes) == 0 and gpu.memory_available > 20480:
            device_ids.append(idx)
    return device_ids


def set_visible_gpus(gpus: int = 1, wait_unitl_available: bool = False) -> None:
    global start
    # gpu_stats = gpustat.new_query()
    # gpu_stats.print_formatted()

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
        print(
            f'using device id {device_ids} at time {strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
    return os.environ['CUDA_VISIBLE_DEVICES']


@click.command()
@click.option('--bash-path', required=True, type=str)
@click.option('--gpus', default=1, type=int)
@click.option('--now', default=False, is_flag=True, type=bool)
def main(bash_path, gpus, now):
    global start
    start = time.time()
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

    subprocess.run(['bash', bash_path], env=os.environ)


if __name__ == "__main__":
    main()
