import math
from collections import Counter, defaultdict

import pandas as pd
import stanza
from matplotlib import pyplot as plt
from tqdm import tqdm


def analysis_pos_distribution():
    en_nlp = stanza.Pipeline('de', processors='tokenize,pos', device='cuda:3', verbose=False,
                             tokenize_pretokenized=True)

    dic = Counter()
    prefix = "/home/nfs01/laizj/data/WMT2023/raw/QE/MQM/en-de/word-level/"
    with open(prefix + "train.src") as src_f, open(prefix + "train.mt") as mt_f, open(
            prefix + "train.tags") as tags_f, tqdm(total=77685) as pbar:
        for s, m_t, t_l in zip(src_f, mt_f, tags_f):
            t_l = t_l.strip().split()
            if t_l[-1] == "BAD":
                dic['<EOS>'] += 1
            t_l = t_l[:-1]
            m_t = en_nlp(m_t).sentences[0]
            for m, t in zip(m_t.words, t_l):
                if t == "BAD":
                    dic[m.pos] += 1
            pbar.update(1)
    df = pd.DataFrame(list(dic.items()))
    print(df)
    df.to_csv('train.csv')


def analysis_mqm_error_prob():
    with open('/Users/dream/Downloads/tmp/dev.k.new') as prob_f, open(
            '/Users/dream/Downloads/tmp/dev.new.tags') as tag_f, open(
        "/Users/dream/Downloads/tmp/dev.zh-en.en.new") as tgt_f:
        prob_f = prob_f.readlines()
        tag_f = tag_f.readlines()
        tgt_f = tgt_f.readlines()

        minor = []
        major = []
        critical = []

        for prob, tgt, tag in zip(prob_f, tgt_f, tag_f):
            prob = prob[1:-1].split(',')
            tag = tag.split()
            tgt = tgt.split()
            pt = 0
            count = 0
            for sub_word, p in zip(tgt, prob):
                if not sub_word.endswith('@@'):
                    if tag[count] == 'minor':
                        minor.append(int(p))
                    elif tag[count] == 'major':
                        major.append(int(p))
                    elif tag[count] == 'critical':
                        critical.append(int(p))
                    pt = 0
                    count += 1

            assert count == len(tag)

        plt.hist([minor, major, critical], label=['minor', 'major', 'critical'])
        # plt.xticks([i for i in range(200)])
        plt.legend()
        plt.show()

        # print(123)
        # print(sum(critical) / len(critical), sum(major) / len(major), sum(minor) / len(minor))

if __name__ == "__main__":
    # stanza.download('en', verbose=False)
    analysis_mqm_error_prob()
