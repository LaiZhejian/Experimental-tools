# adapted from https://github.com/oobabooga/text-generation-webui/blob/main/download-model.py

"""
Downloads models from Hugging Face to ./username_modelname.

Example:
python download-model.py facebook/opt-1.3b

"""

# ADDON: connection fix start

from tqdm.contrib.concurrent import thread_map
from requests.adapters import HTTPAdapter
import tqdm
import requests
from pathlib import Path
import sys
import re
import os
import json
import hashlib
import datetime
import base64
import argparse
import urllib3
from urllib3.util import connection
import urllib3.contrib.pyopenssl
from OpenSSL import SSL

SSL.Connection.set_tlsext_host_name = lambda *args, **kwargs: None


urllib3.contrib.pyopenssl.inject_into_urllib3()


_orig_create_connection = connection.create_connection


def _dns_resolver(host: str):
    if host.endswith('huggingface.co'):
        return '13.224.157.70'
    else:
        return host


def patched_create_connection(address, *args, **kwargs):
    host, port = address
    hostname = _dns_resolver(host)
    return _orig_create_connection((hostname, port), *args, **kwargs)


connection.create_connection = patched_create_connection


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# connection fix end


base = "https://huggingface.co"


class ModelDownloader:
    def __init__(self, max_retries=5):
        self.session = requests.Session()
        self.session.verify = False  # ADDON: disable certificate verification
        if max_retries:
            self.session.mount('https://cdn-lfs-us-1.huggingface.co', HTTPAdapter(max_retries=max_retries))
            self.session.mount('https://cdn-lfs.huggingface.co', HTTPAdapter(max_retries=max_retries))
            self.session.mount('https://huggingface.co', HTTPAdapter(max_retries=max_retries))
        if os.getenv('HF_USER') is not None and os.getenv('HF_PASS') is not None:
            self.session.auth = (os.getenv('HF_USER'), os.getenv('HF_PASS'))
        if os.getenv('HF_TOKEN') is not None:
            self.session.headers = {'authorization': f'Bearer {os.getenv("HF_TOKEN")}'}

    def sanitize_model_and_branch_names(self, model, branch):
        if model[-1] == '/':
            model = model[:-1]

        if model.startswith(base + '/'):
            model = model[len(base) + 1:]

        model_parts = model.split(":")
        model = model_parts[0] if len(model_parts) > 0 else model
        branch = model_parts[1] if len(model_parts) > 1 else branch

        if branch is None:
            branch = "main"
        else:
            pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
            if not pattern.match(branch):
                raise ValueError(
                    "Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed.")

        return model, branch

    def get_download_links_from_huggingface(self, model, branch, text_only=False, specific_file=None):
        page = f"/api/models/{model}/tree/{branch}"
        cursor = b""

        links = []
        sha256 = []
        classifications = []
        has_pytorch = False
        has_pt = False
        has_gguf = False
        has_safetensors = False
        is_lora = False
        while True:
            url = f"{base}{page}" + (f"?cursor={cursor.decode()}" if cursor else "")
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            content = r.content

            dict = json.loads(content)
            if len(dict) == 0:
                break

            for i in range(len(dict)):
                fname = dict[i]['path']
                if specific_file not in [None, ''] and fname != specific_file:
                    continue

                if not is_lora and fname.endswith(('adapter_config.json', 'adapter_model.bin')):
                    is_lora = True

                is_pytorch = re.match(r"(pytorch|adapter|gptq)_model.*\.bin", fname)
                is_safetensors = re.match(r".*\.safetensors", fname)
                is_pt = re.match(r".*\.pt", fname)
                is_gguf = re.match(r'.*\.gguf', fname)
                is_tiktoken = re.match(r".*\.tiktoken", fname)
                is_tokenizer = re.match(r"(tokenizer|ice|spiece).*\.model", fname) or is_tiktoken
                is_text = re.match(r".*\.(txt|json|py|md)", fname) or is_tokenizer
                if any((is_pytorch, is_safetensors, is_pt, is_gguf, is_tokenizer, is_text)):
                    if 'lfs' in dict[i]:
                        sha256.append([fname, dict[i]['lfs']['oid']])

                    if is_text:
                        links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
                        classifications.append('text')
                        continue

                    if not text_only:
                        links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
                        if is_safetensors:
                            has_safetensors = True
                            classifications.append('safetensors')
                        elif is_pytorch:
                            has_pytorch = True
                            classifications.append('pytorch')
                        elif is_pt:
                            has_pt = True
                            classifications.append('pt')
                        elif is_gguf:
                            has_gguf = True
                            classifications.append('gguf')

            cursor = base64.b64encode(f'{{"file_name":"{dict[-1]["path"]}"}}'.encode()) + b':50'
            cursor = base64.b64encode(cursor)
            cursor = cursor.replace(b'=', b'%3D')

        # If both pytorch and safetensors are available, download safetensors only
        if (has_pytorch or has_pt) and has_safetensors:
            for i in range(len(classifications) - 1, -1, -1):
                if classifications[i] in ['pytorch', 'pt']:
                    links.pop(i)

        is_llamacpp = has_gguf and specific_file is not None
        return links, sha256, is_lora, is_llamacpp

    def get_output_folder(self, model, branch, is_lora, is_llamacpp=False, base_folder=None):
        if base_folder is None:
            base_folder = './'

        # If the model is of type GGUF, save directly in the base_folder
        if is_llamacpp:
            return Path(base_folder)

        output_folder = f"{'_'.join(model.split('/')[-2:])}"
        if branch != 'main':
            output_folder += f'_{branch}'

        output_folder = Path(base_folder) / output_folder
        return output_folder

    def get_single_file(self, url, output_folder, start_from_scratch=False):
        filename = Path(url.rsplit('/', 1)[1])
        output_path = output_folder / filename
        headers = {}
        mode = 'wb'
        if output_path.exists() and not start_from_scratch:

            # Check if the file has already been downloaded completely
            r = self.session.get(url, stream=True, timeout=10)
            total_size = int(r.headers.get('content-length', 0))
            if output_path.stat().st_size >= total_size:
                return

            # Otherwise, resume the download from where it left off
            headers = {'Range': f'bytes={output_path.stat().st_size}-'}
            mode = 'ab'

        with self.session.get(url, stream=True, headers=headers, timeout=10) as r:
            r.raise_for_status()  # Do not continue the download if the request was unsuccessful
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB
            with open(output_path, mode) as f:
                with tqdm.tqdm(total=total_size, unit='iB', unit_scale=True,
                               bar_format='{l_bar}{bar}| {n_fmt:6}/{total_fmt:6} {rate_fmt:6} [{elapsed}<{remaining}]') as t:
                    count = 0
                    for data in r.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)
                        if total_size != 0 and self.progress_bar is not None:
                            count += len(data)
                            self.progress_bar(float(count) / float(total_size), f"{filename}")

    def start_download_threads(self, file_list, output_folder, start_from_scratch=False, threads=1):
        if threads == 1:
            for url in file_list:
                self.get_single_file(url, output_folder, start_from_scratch=start_from_scratch)
        else:
            thread_map(lambda url: self.get_single_file(url, output_folder, start_from_scratch=start_from_scratch),
                       file_list, max_workers=threads, disable=True)

    def download_model_files(self, model, branch, links, sha256, output_folder, progress_bar=None,
                             start_from_scratch=False, threads=1, specific_file=None, is_llamacpp=False):
        self.progress_bar = progress_bar

        # Create the folder and writing the metadata
        output_folder.mkdir(parents=True, exist_ok=True)

        if not is_llamacpp:
            metadata = f'url: https://huggingface.co/{model}\n' \
                       f'branch: {branch}\n' \
                       f'download date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'

            sha256_str = '\n'.join([f'    {item[1]} {item[0]}' for item in sha256])
            if sha256_str:
                metadata += f'sha256sum:\n{sha256_str}'

            metadata += '\n'
            (output_folder / 'huggingface-metadata.txt').write_text(metadata)

        if specific_file:
            print(f"Downloading {specific_file} to {output_folder}")
        else:
            print(f"Downloading the model to {output_folder}")

        self.start_download_threads(links, output_folder, start_from_scratch=start_from_scratch, threads=threads)

    def check_model_files(self, model, branch, links, sha256, output_folder):
        # Validate the checksums
        validated = True
        for i in range(len(sha256)):
            fpath = (output_folder / sha256[i][0])

            if not fpath.exists():
                print(f"The following file is missing: {fpath}")
                validated = False
                continue

            with open(output_folder / sha256[i][0], "rb") as f:
                bytes = f.read()
                file_hash = hashlib.sha256(bytes).hexdigest()
                if file_hash != sha256[i][1]:
                    print(f'Checksum failed: {sha256[i][0]}  {sha256[i][1]}')
                    validated = False
                else:
                    print(f'Checksum validated: {sha256[i][0]}  {sha256[i][1]}')

        if validated:
            print('[+] Validated checksums of all model files!')
        else:
            print('[-] Invalid checksums. Rerun download-model.py with the --clean flag.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL', type=str, default=None, nargs='?')
    parser.add_argument('--branch', type=str, default='main', help='Name of the Git branch to download from.')
    parser.add_argument('--threads', type=int, default=1, help='Number of files to download simultaneously.')
    parser.add_argument('--text-only', action='store_true', help='Only download text files (txt/json).')
    parser.add_argument('--specific-file', type=str, default=None,
                        help='Name of the specific file to download (if not provided, downloads all).')
    parser.add_argument('--output', type=str, default=None, help='The folder where the model should be saved.')
    parser.add_argument('--clean', action='store_true', help='Does not resume the previous download.')
    parser.add_argument('--check', action='store_true', help='Validates the checksums of model files.')
    parser.add_argument('--max-retries', type=int, default=5, help='Max retries count when get error in download time.')
    args = parser.parse_args()

    branch = args.branch
    model = args.MODEL
    specific_file = args.specific_file

    if model is None:
        print(
            "Error: Please specify the model you'd like to download (e.g. 'python download-model.py facebook/opt-1.3b').")
        sys.exit()

    downloader = ModelDownloader(max_retries=args.max_retries)
    # Clean up the model/branch names
    try:
        model, branch = downloader.sanitize_model_and_branch_names(model, branch)
    except ValueError as err_branch:
        print(f"Error: {err_branch}")
        sys.exit()

    # Get the download links from Hugging Face
    links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch,
                                                                                         text_only=args.text_only,
                                                                                         specific_file=specific_file)

    # Get the output folder
    output_folder = downloader.get_output_folder(model, branch, is_lora, is_llamacpp=is_llamacpp,
                                                 base_folder=args.output)

    if args.check:
        # Check previously downloaded files
        downloader.check_model_files(model, branch, links, sha256, output_folder)
    else:
        # Download files
        downloader.download_model_files(model, branch, links, sha256, output_folder, specific_file=specific_file,
                                        threads=args.threads, is_llamacpp=is_llamacpp)
