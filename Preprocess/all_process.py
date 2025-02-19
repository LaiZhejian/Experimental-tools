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
mtr = {src_lang: "/home/data_91_c/laizj/data/WMT/23/parallel/wmt23-ende/train.en-de.en.truecasemodel", tgt_lang: "/home/data_91_c/laizj/data/WMT/23/parallel/wmt23-ende/train.en-de.de.truecasemodel"}
read_file_1 = f"/home/data_91_c/laizj/data/WMT/23/parallel/wmt23-ende/total/eng_Latn.devtest"
read_file_2 = f"/home/data_91_c/laizj/data/WMT/23/parallel/wmt23-ende/total/deu_Latn.devtest"
output_file_1 = f"/home/data_91_c/laizj/data/WMT/23/parallel/wmt23-ende/total/test.en-de.tok.en"
output_file_2 = f"/home/data_91_c/laizj/data/WMT/23/parallel/wmt23-ende/total/test.en-de.tok.de"


mpn = sacremoses.MosesPunctNormalizer()
mtr = {src_lang: sacremoses.MosesTruecaser(mtr[src_lang]) if mtr[src_lang] else None,tgt_lang: sacremoses.MosesTruecaser(mtr[tgt_lang]) if mtr[tgt_lang] else None}
mt = {src_lang: sacremoses.MosesTokenizer(lang=src_lang), tgt_lang: sacremoses.MosesTokenizer(lang=tgt_lang)}


def tokenize(line, lang):
    line = html.unescape(line)
    return mt[lang].tokenize(line, escape=False, return_str=True)

def truecase(line, lang):
    return mtr[lang].truecase(line, return_str=True)

if __name__ == "__main__":
    with open(read_file_1) as rf1, open(read_file_2) as rf2: 
        with open(output_file_1, "w") as of1, open(output_file_2, "w") as of2:
            for line1, line2 in zip(rf1, rf2):
                line1 = line1.strip()
                line2 = line2.strip()
                line1 = mpn.normalize(line1)
                line2 = mpn.normalize(line2)
                line1 = tokenize(line1, src_lang)
                line2 = tokenize(line2, tgt_lang)
                line1 = truecase(line1, src_lang)
                line2 = truecase(line2, tgt_lang)
                of1.write(line1 + "\n")
                of2.write(line2 + "\n")
            