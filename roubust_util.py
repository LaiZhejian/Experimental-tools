import numpy as np
from tqdm import tqdm
import logging
import os
import sys
from f1_mcc_pearson.f1cal_mcc import precision_recall_fscore_support

if __name__ == "__main__":

    _y_hat_path = "/home/gengx/laizj/ft-xlmr/test.tags"
    _y_path = "/home/data_91_c/laizj/data/Challenge-Set/Yanym-cons/robust_qe_test_enzh/generate_dev/robust_test.tag"
    _idx_path = "/home/data_91_c/laizj/data/Challenge-Set/Yanym-cons/robust_qe_test_enzh/generate_dev/robust_test.idx"

    _y_hat = []
    with open(_y_hat_path) as f:
        for line in f:
            _y_hat.extend(line.strip().split())

    y = []
    y_hat = []
    with open(_y_path) as f1, open(_idx_path) as f2:
        count = 0
        for line, idx_line in zip(f1, f2):
            line = line.strip().split()
            for idx in idx_line.strip().split():
                y_hat.append(int(_y_hat[count + int(idx)] == "BAD"))
                y.append(int(line[int(idx)] == "BAD"))
            count += len(line)

    assert count == len(_y_hat)
    _, _, f1, _, mcc = precision_recall_fscore_support(y, y_hat)
    print(f'OK:{f1[0]}, BAD:{f1[1]}')
    print(f'F1-Mult:{f1[0]*f1[1]}')
    print(f'MCC:{mcc[0]}')


