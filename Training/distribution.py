import datetime
from time import gmtime, sleep, strftime
import time
import click
import os
import subprocess
import gpustat
import fcntl



def get_available_devices():
    device_ids = []
    gpu_stats = gpustat.new_query()
    for idx, gpu in enumerate(gpu_stats):
        if len(gpu.processes) == 0 and gpu.memory_available > 20480:
            device_ids.append(idx)
    return device_ids


def set_visible_gpus(gpus: int = 1, wait_unitl_available: bool = False) -> None:
    # gpu_stats = gpustat.new_query()
    # gpu_stats.print_formatted()

    device_ids = get_available_devices()
    device_ids = device_ids[:gpus]
    if not wait_unitl_available:
        print(
            f'using device id {device_ids} at time {strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
    return os.environ['CUDA_VISIBLE_DEVICES']


@click.command()
@click.option('--bash-path', required=True, type=str)
@click.option('--gpus', default=1, type=int)
def main(bash_path, gpus):
    
    avai_cuda_devices = set_visible_gpus(gpus=gpus)
    if not avai_cuda_devices:
        exit(1)

    os.execve('/bin/bash', ['/bin/bash', bash_path], os.environ)


if __name__ == "__main__":
    main()
