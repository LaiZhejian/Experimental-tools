import os
from time import sleep
import gpustat


def set_visible_gpus(gpus: int = 1, wait_unitl_available: bool = False) -> None:
    gpu_stats = gpustat.new_query()
    # gpu_stats.print_formatted()
    # print('waiting for gpu...')
    while True:
        device_ids = []
        for idx, gpu in enumerate(gpu_stats):
            if len(gpu.processes) == 0 or gpu.memory_used < 20:
                device_ids.append(idx)
        if len(device_ids) >= gpus:
            break
        else:
            if wait_unitl_available:
                sleep(1)
                gpu_stats = gpustat.new_query()
            else:
                raise RuntimeError(f'No enough gpu! {gpus} is required, but only {len(device_ids)} is left')
    device_ids = device_ids[:gpus]
    # print(f'using device id {device_ids}')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
    return os.environ['CUDA_VISIBLE_DEVICES']


if __name__ == '__main__':
    print(set_visible_gpus())
