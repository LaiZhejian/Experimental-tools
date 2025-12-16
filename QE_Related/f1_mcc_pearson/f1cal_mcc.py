# sys and gold have the same length, but the gold file has pad tag

import numpy as np
import pandas as pd
import click


# prefix = "ende"
# file = "dev"

def precision_recall_fscore_support(y, y_hat, n_classes=2):
    confusion_matrix = np.zeros((n_classes, n_classes))
    for j in range(len(y)):
        confusion_matrix[y[j], y_hat[j]] += 1

    scores = np.zeros((n_classes, 5))
    for class_id in range(n_classes):
        scores[class_id] = scores_for_class(class_id, confusion_matrix)
    return scores.T.tolist()


def scores_for_class(class_index, matrix):
    tp = matrix[class_index, class_index]
    fp = matrix[:, class_index].sum() - tp
    fn = matrix[class_index, :].sum() - tp
    tn = matrix.sum() - tp - fp - fn

    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = mcc_numerator / mcc_denominator

    p, r, f1 = fscore(tp, fp, fn)
    support = tp + tn
    return p, r, f1, support, mcc


def precision(tp, fp):
    if tp + fp > 0:
        return tp / (tp + fp)
    return 0


def recall(tp, fn):
    if tp + fn > 0:
        return tp / (tp + fn)
    return 0


def fscore(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    if p + r > 0:
        return p, r, 2 * (p * r) / (p + r)
    return p, r, 0


def word_level(system_word_file, gold_word_file):

    system_tag = []
    gold_tag = []

    # 计算MCC与F1 Score
    with open(system_word_file) as f1, open(gold_word_file) as f2:
        f1 = f1.readlines()
        f2 = f2.readlines()
        assert len(f1) == len(f2)
        for line1, line2 in zip(f1, f2):
            words1, words2 = line1.strip().split(), line2.strip().split()
            assert len(words1) == len(words2) or len(
                words1) * 2 + 1 == len(words2)
            if len(words1) != len(words2):
                words2 = words2[1:: 2]
            for word in words1:
                if word == "BAD":
                    system_tag.append(1)
                else:
                    system_tag.append(0)
            for word in words2:
                if word == "BAD":
                    gold_tag.append(1)
                elif word == "OK":
                    gold_tag.append(0)

    y = []
    y_hat = []
    for i in range(len(gold_tag)):
        if gold_tag[i] != -1:
            y.append(gold_tag[i])
            y_hat.append(system_tag[i])

    _, _, f1, _, mcc = precision_recall_fscore_support(y, y_hat)
    print(f'OK:{f1[0]}, BAD:{f1[1]}')
    print(f'F1-Mult:{f1[0]*f1[1]}')
    print(f'MCC:{mcc[0]}')


def sent_level(system_sent_file, gold_sent_file):

    with open(system_sent_file) as f1, open(gold_sent_file) as f2:

        f1 = pd.Series([float(x.strip()) for x in f1])
        f2 = pd.Series([float(x.strip()) for x in f2])
        print(f'Pearson:{f1.corr(f2)}')
        print(f'Spearman: {f1.corr(f2, method="spearman")}')


@click.command()
@click.option('--system-sent-file', type=click.Path(), help='System file path', default="")
@click.option('--gold-sent-file', type=click.Path(), help='Gold file path', default="")
@click.option('--system-word-file', type=click.Path(), help='System file path', default="")
@click.option('--gold-word-file', type=click.Path(), help='Gold file path', default="")
def main(system_sent_file, gold_sent_file, system_word_file, gold_word_file):

    print(system_sent_file, gold_sent_file, system_word_file, gold_word_file)
    if system_sent_file != "" and gold_sent_file != "":
        sent_level(system_sent_file, gold_sent_file)
    if system_word_file != "" and gold_word_file != "":
        word_level(system_word_file, gold_word_file)


if __name__ == "__main__":
    main()
    print("finished")
