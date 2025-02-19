import argparse
import sys
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from statsmodels.stats.multitest import multipletests
from tqdm.contrib.concurrent import process_map

def read_word_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split() for line in lines]


def read_sent_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [float(line.strip()) for line in lines]

def randomization_iteration(flatten_gold, data1, data2):

    new_preds1 = []
    new_preds2 = []
    swap_mask = np.random.rand(len(data1)) > 0.5
    for idx, mask in enumerate(swap_mask):
        if mask:
            new_preds1 = new_preds1 + data1[idx]
            new_preds2 = new_preds2 + data2[idx]
        else:
            new_preds1 = new_preds1 + data2[idx]
            new_preds2 = new_preds2 + data1[idx]

    # new_preds1 = np.where(swap_mask, data1, data2)
    # new_preds2 = np.where(swap_mask, data2, data1)

    mcc_new_group1 = matthews_corrcoef(flatten_gold, new_preds1)
    mcc_new_group2 = matthews_corrcoef(flatten_gold, new_preds2)

    return abs(mcc_new_group1 - mcc_new_group2)

def randomisation_test(gold, data1, data2, num_iterations=100):
    """Perform a randomization test."""


    def flatten(data):
        return [word for line in data for word in line]

    flatten_gold = flatten(gold)

    # 计算MCC
    mcc_group1 = matthews_corrcoef(flatten_gold, flatten(data1))
    mcc_group2 = matthews_corrcoef(flatten_gold, flatten(data2))

    # 定义检验统计量
    delta_mcc = abs(mcc_group1 - mcc_group2)

    
    delta_mcc_random = process_map(randomization_iteration, 
                                    [flatten_gold] * num_iterations,
                                    [data1] * num_iterations,
                                    [data2] * num_iterations,
                                    max_workers=32, disable=True,
                                    total=num_iterations, chunksize=3)
    
    # delta_mcc_random = []
    
    # for _ in range(num_iterations):

    #     swap_mask = np.random.rand(len(data1)) > 0.5
    #     new_preds1 = np.where(swap_mask, data1, data2)
    #     new_preds2 = np.where(swap_mask, data2, data1)

    #     mcc_new_group1 = matthews_corrcoef(gold, new_preds1)
    #     mcc_new_group2 = matthews_corrcoef(gold, new_preds2)
    #     delta_mcc_random.append(abs(mcc_new_group1 - mcc_new_group2))


    delta_mcc_random = np.array(delta_mcc_random)

    # 计算p值
    p_value = np.sum(delta_mcc_random >= delta_mcc) / num_iterations
    return p_value


def williams_test(r12, r13, r23, n):
    """The Williams test (Evan J. Williams. 1959. Regression Analysis, volume 14. Wiley, New York, USA)
    
    A test of whether the population correlation r12 equals the population correlation r13.
    Significant: p < 0.05
    
    Arguments:
        r12 (float): correlation between x1, x2
        r13 (float): correlation between x1, x3
        r23 (float): correlation between x2, x3
        n (int): size of the population
        
    Returns:
        t (float): Williams test result
        p (float): p-value of t-dist
    """
    if r12 < r13:
        print('r12 should be larger than r13')
        sys.exit()
    elif n <= 3:
        print('n should be larger than 3')
        sys.exit()
    else:
        K = 1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23
        denominator = np.sqrt(
            2*K*(n-1)/(n-3) + (((r12+r13)**2)/4)*((1-r23)**3))
        numerator = (r12-r13) * np.sqrt((n-1)*(1+r23))
        t = numerator / denominator
        p = 1 - stats.t.cdf(t, df=n-3)  # changed to n-3 on 30/11/14
        return t, p


def sent_significance_test(gold_data, data1, data2):
    
    n = len(gold_data)
    r12 = spearmanr(gold_data, data1)[0]
    r13 = spearmanr(gold_data, data2)[0]
    r23 = spearmanr(data1, data2)[0]

    _, p_value = williams_test(max(r12, r13), min(r12, r13), r23, n)
    
    return p_value

def main(gold_word_file, word_file_1, word_file_2, gold_sent_file, sent_file_1, sent_file_2):
    
    if sent_file_1:
        # Read sentence files
        sent_data_1 = read_sent_file(sent_file_1)
        sent_data_2 = read_sent_file(sent_file_2)
        # del(sent_data_2[983])
        # del(sent_data_2[647])
        gold_data = read_sent_file(gold_sent_file)

        # Perform William's test
        p_value_sent = sent_significance_test(
            gold_data, sent_data_1, sent_data_2)

        print("=" * 10 + "SENT-LEVEL" + "=" * 10)
        print(f"William's Test p-value: {p_value_sent:.4f}")
    
    if word_file_1:
        # Read word files
        word_data_1 = read_word_file(word_file_1)
        word_data_2 = read_word_file(word_file_2)
        # del (word_data_2[983])
        # del (word_data_2[647])
        gold_data = read_word_file(gold_word_file)
        # Perform randomisation test

        test_nums = 5
        p_values = []
        for i in range(test_nums):
            p_values.append(randomisation_test(gold_data, word_data_1, word_data_2))
    
        # Apply Bonferroni correction
        p_value_word_corrected = multipletests(
            p_values, alpha=0.05, method='bonferroni')[1][0]

        print("=" * 10 + "WORD-LEVEL" + "=" * 10)
        print(
            f"Randomisation Test p-value (Bonferroni corrected): {p_value_word_corrected:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform randomisation tests with Bonferroni correction and William’s test.')
    parser.add_argument('--gold-word-file', default=None,
                        help='Gold word file path')
    parser.add_argument('--word-file1', default=None,
                        help='First word file path')
    parser.add_argument('--word-file2', default=None,
                        help='Second word file path')
    parser.add_argument('--gold-sent-file', default=None,
                        help='Gold sentence file path')
    parser.add_argument('--sent-file1', default=None,
                        help='First sentence file path')
    parser.add_argument('--sent-file2', default=None,
                        help='Second sentence file path')

    args = parser.parse_args()
    main(args.gold_word_file, args.word_file1, args.word_file2,
         args.gold_sent_file, args.sent_file1, args.sent_file2)
