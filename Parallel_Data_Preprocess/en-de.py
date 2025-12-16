import os
import numpy as np
import pandas as pd
import json
import sacremoses
from tqdm import tqdm
from langconv import *
import html
import langid

src_lang = "en"
tgt_lang = "de"
prefix = f"train.{src_lang}-{tgt_lang}"
file1 = f"/home/nfs01/laizj/data/WMT2023/raw/parallel/wmt23-ende/train.{src_lang}-{tgt_lang}.{src_lang}"
file2 = f"/home/nfs01/laizj/data/WMT2023/raw/parallel/wmt23-ende/train.{src_lang}-{tgt_lang}.{tgt_lang}"

def get_file_lines(fname):
    count = 0
    with open(fname) as f:
        for _ in f:
            count += 1
    return count


if __name__ == "__main__":

    mpn = sacremoses.MosesPunctNormalizer()
    mtr = {src_lang: sacremoses.MosesTruecaser(
        f"{file1}.truecasemodel" if os.path.exists(f"{file1}.truecasemodel") else None), \
        tgt_lang: sacremoses.MosesTruecaser(
            f"{file2}.truecasemodel" if os.path.exists(f"{file2}.truecasemodel") else None)}
    mt = {src_lang: sacremoses.MosesTokenizer(lang=src_lang), tgt_lang: sacremoses.MosesTokenizer(lang=tgt_lang)}

    # with open(file1) as src_f, open(file2) as tgt_f:
    #     print("开始处理字符：去除UTF-8，HTML转义, Normalize， Tokenize")
    #     count = get_file_lines(file1)
    #     src_f = src_f.readlines()
    #     tgt_f = tgt_f.readlines()
    #     for f, fn in ((src_f, src_lang), (tgt_f, tgt_lang)):
    #         process_bar = tqdm(total=count)
    #         for idx, line in enumerate(f):
    #             # 去除非UTF-8
    #             line = line.encode(
    #                 'utf-8', errors='ignore'
    #             ).decode('utf-8')
    #             # HTML转义
    #             line = html.unescape(line)
    #             # Normalize
    #             line = mpn.normalize(line)
    #             # Tokenize
    #             line = mt[fn].tokenize(line, escape=False, return_str=True)

    #             f[idx] = line
    #             process_bar.update(1)
    #     print(len(src_f))

    # print("开始处理字符：第一次去重")
    # dic = {
    #     'src': pd.Series(src_f),
    #     'tgt': pd.Series(tgt_f)
    # }
    # df = pd.DataFrame(dic)
    # df = df.drop_duplicates(subset='src').drop_duplicates(subset='tgt')
    # src_f = df['src'].to_list()
    # tgt_f = df['tgt'].to_list()
    # print(len(src_f))

    # with open(file1 + ".1", "w") as f_o1, open(file2 + ".1", "w") as f_o2:
    #     fo = {src_lang: f_o1, tgt_lang: f_o2}
    #     for f, fn in ((src_f, src_lang), (tgt_f, tgt_lang)):
    #         for idx, line in enumerate(f):
    #             fo[fn].write(line + "\n")

    # with open(file3 + ".1") as src_f, open(file4 + ".1") as tgt_f:
    #     count = get_file_lines(file1 + ".1")
    #     src_f = src_f.readlines()
    #     tgt_f = tgt_f.readlines()

    #     print("开始处理字符：训练Truecase")

    #     for fn, fp in ((src_lang, file1), (tgt_lang, file2)):
    #         mtr[fn].train(src_f, save_to=f"{fp}.truecasemodel", progress_bar=True)

        
    final_count = 0
    print(
        "开始处理字符：Truecase, language error, src==tgt， blank line， too long sentence, abnormal src-tgt length ratio,\
         irregular character-word length ratios, contain too long words")

    with open(file1 + ".1") as src_f, open(file2 + ".1") as tgt_f, open(
        f"train.{src_lang}-{tgt_lang}.{src_lang}.final", "w") as f_o1, open(
        f"train.{src_lang}-{tgt_lang}.{tgt_lang}.final", "w") as f_o2:

        count = get_file_lines(file1 + ".1")
        src_f = src_f.readlines()
        tgt_f = tgt_f.readlines()     
        process_bar = tqdm(total=count)
        for idx, (line_s, line_t) in enumerate(zip(src_f, tgt_f)):
            process_bar.update(1)
            line_s = mtr[src_lang].truecase(line_s, return_str=True)
            line_t = mtr[tgt_lang].truecase(line_t, return_str=True)

            # filter src!=zh  tgt!=en
            if langid.classify(line_s)[0] != src_lang or langid.classify(line_t)[0] != tgt_lang: continue
            # filter src==tgt
            if line_s == line_t: continue
            # filter blankline
            if line_s == "" or line_t == "": continue

            line_s_tokens = line_s.split()
            line_t_tokens = line_t.split()
            len_s = len(line_s_tokens)
            len_t = len(line_t_tokens)
            # filter too long sentence
            if len_s > 200 or len_t > 200: continue
            # filter abnormal src-tgt length ratio
            if 2.5 < len_s / len_t or 0.4 > len_s / len_t: continue
            # filter irregular character-word length ratios
            sum_tokens_s = sum(map(len, line_s_tokens))
            sum_tokens_t = sum(map(len, line_t_tokens))
            if sum_tokens_s / len_s > 12 or sum_tokens_s / len_s < 1.5: continue
            if sum_tokens_t / len_t > 12 or sum_tokens_t / len_t < 1.5: continue
            # filter contain too long words
            max_len_token_s = max(map(len, line_s_tokens))
            max_len_token_t = max(map(len, line_t_tokens))
            if max_len_token_s > 20 or max_len_token_t > 20: continue

            f_o1.write(line_s + "\n")
            f_o2.write(line_t + "\n")
            final_count += 1

        print(final_count)
