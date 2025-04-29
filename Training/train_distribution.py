import datetime
import time
import sys
import os

start = time.time()
DEFAULT_MUTEX_PATH = "/tmp/laizj-ysqd-mutex.lock"

hosts = ["A6000-1", "A6000-2", "A6000-3", "A6000-4", "A6000-5", "A6000-6"]

def main():
    args = sys.argv[1:]
    args = (*args, "--no_wait")
    while True:
        for host in hosts:
            
            print(f"\n\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking server: {host}...")
            command = f"ssh -o StrictHostKeyChecking=no {host} 'source /home/nfs03/laizj/envs/laizj-torch2.5-py310; ysqd {' '.join(args)}'"
            return_value = os.WEXITSTATUS(os.system(command))
            time.sleep(2)
            
            if return_value != 255:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Command succeeded on {host}, return value: {return_value}")
                return 0
            else:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Command failed on {host}, retrying in 5 seconds...")
        

if __name__ == "__main__":
    main()
