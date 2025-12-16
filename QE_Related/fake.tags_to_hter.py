import random

import numpy as np
from tqdm import tqdm
import logging
import os
import sys


def make_hter_from_tags():
    file_path1 = "/home/data_91_c/laizj/data/Challenge-Set/Mytest/sentencepiece/test.tags-None.tags"
    t_output_path1 = "/home/data_91_c/laizj/data/Challenge-Set/Mytest/sentencepiece/test.hter"

    count = 0
    with open(file_path1) as fp1:
        for _ in fp1:
            count += 1

    with open(file_path1) as fp1, open(t_output_path1, "w") as op1, tqdm(total=count) as pbar:

        for line in fp1:
            count = 0
            line = line.strip().split()
            for tag in line:
                if tag == 'BAD':
                    count += 1
            op1.write('{}\n'.format(count / len(line)))
            pbar.update(1)


def make_fake_hter_tag():
    file_target = "/home/data_91_c/laizj/data/tmp/test.en-zh.zh"
    file_path_t = "/home/data_91_c/laizj/data/tmp/test.tags-None.tags"
    file_path_h = "/home/data_91_c/laizj/data/tmp/test.hter"
    with open(file_target) as tgt_f, open(file_path_t, 'w') as tag_f, open(file_path_h, "w") as score_f:
        for line in tgt_f:
            line = line.split()
            tags = ""
            for i in range(len(line)):
                tags += "OK " if random.random() > 0.5 else "BAD "
            tag_f.write(tags + "\n")
            score_f.write(str(random.random()) + "\n")


if __name__ == "__main__":
    make_fake_hter_tag()
