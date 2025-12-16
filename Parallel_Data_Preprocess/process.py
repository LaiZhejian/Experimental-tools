import os

from tqdm import tqdm

from langconv import *
import html

SRC_LANG = "en"
TGT_LANG = "zh"
SPLIT = "dev"

SCRIPTS = f"/home/data_91_c/laizj/code/utils/mosesdecoder/scripts"
TOKENIZER = f"{SCRIPTS}/tokenizer/tokenizer.perl"
DETOKENIZER = f"{SCRIPTS}/tokenizer/detokenizer.perl"
LC = f"{SCRIPTS}/tokenizer/lowercase.perl"
TRAIN_TC = f"{SCRIPTS}/recaser/train-truecaser.perl"
TC = f"{SCRIPTS}/recaser/truecase.perl"
DETC = f"{SCRIPTS}/recaser/detruecase.perl"
NORM_PUNC = f"{SCRIPTS}/tokenizer/normalize-punctuation.perl"
CLEAN = f"{SCRIPTS}/training/clean-corpus-n.perl"
BPEROOT = f"/home/data_91_c/laizj/utils/subword-nmt/subword_nmt"
MULTI_BLEU = f"{SCRIPTS}/generic/multi-bleu.perl"
MTEVAL_V14 = f"{SCRIPTS}/generic/mteval-v14.pl"

data_dir = "/home/data_91_c/laizj/data/CCMT/Pretrain-NMT"
model_dir = "/home/data_91_c/laizj/data/CCMT/Pretrain-NMT/models"
dev_dir = '/home/data_91_c/laizj/data/CCMT/mteval.cipsc.org.cn/CCMT&WMT2023.ch-en/dev/CCMT2021EC/dev'


def half_to_full(text: str):
    _text = ""
    for char in text:
        inside_code = ord(char)
        if inside_code == 32:
            inside_code = 12288
        elif 33 <= inside_code <= 132:
            inside_code += 65248
        _text += chr(inside_code)
    return _text


def full_to_half(text: str):  # 输入为一个句子
    _text = ""
    for char in text:
        inside_code = ord(char)  # 以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        _text += chr(inside_code)
    return _text


if __name__ == "__main__":

    for LANG in [SRC_LANG, TGT_LANG]:
        print("开始处理" + LANG)
        path = f"/home/data_91_c/laizj/data/CCMT/Pretrain-NMT/{SPLIT}.raw.{LANG}"
        output_path = f"/home/data_91_c/laizj/data/CCMT/Pretrain-NMT/{SPLIT}.init.{LANG}"

        tradition2Simplifiedconverter = Converter('zh-hans')
        count = 0
        with open(path) as f:
            for line in f:
                count += 1

        print("开始处理字符：去除UTF-8，HTML转义")
        process_bar = tqdm(total=count)
        with open(path) as f, open(output_path, "w") as fo:
            for line in f:
                # 去除非UTF-8
                line = line.encode(
                    'utf-8', errors='ignore'
                ).decode('utf-8')
                # 将HTML转义字符变为文本
                if LANG == "zh":
                    # 转化为简体中文
                    line = tradition2Simplifiedconverter.convert(line)
                    # 转化为半角
                    line = full_to_half(line)
                fo.write(line.strip() + "\n")
                process_bar.update(1)

        # Normalization
        print("开始处理字符：Normalization")
        os.system(f"perl {NORM_PUNC} -l {LANG} < {data_dir}/{SPLIT}.init.{LANG} > {data_dir}/{SPLIT}.norm.{LANG}")

        # 中文分词
        if LANG == "zh":
            print("开始处理字符：中文分词")
            import pkuseg

            seg = pkuseg.pkuseg()
            process_bar = tqdm(total=count)
            path = f"{data_dir}/{SPLIT}.norm.zh"
            output_path = f"{data_dir}/{SPLIT}.norm.seg.zh"
            with open(path) as f, open(output_path, "w") as fo:
                for line in f:
                    line = seg.cut(line)
                    fo.write(" ".join(line) + "\n")
                    process_bar.update(1)
            os.system(f"mv {data_dir}/{SPLIT}.norm.seg.zh {data_dir}/{SPLIT}.norm.zh")

        # Tokenization
        print("开始处理字符：Tokenization")
        os.system(
            f"perl {TOKENIZER} -no-escape -threads 80 -a -l {LANG} < {data_dir}/{SPLIT}.norm.{LANG} > {data_dir}/{SPLIT}.norm.tok.{LANG}")

    # 去重以及去除含有非英文字符的语言对
    if SPLIT == "train":
        src_path = f"{data_dir}/{SPLIT}.norm.tok.{SRC_LANG}"
        tgt_path = f"{data_dir}/{SPLIT}.norm.tok.{TGT_LANG}"
        src_output_path = f"{data_dir}/{SPLIT}.norm.tok.dedup.{SRC_LANG}"
        tgt_output_path = f"{data_dir}/{SPLIT}.norm.tok.dedup.{TGT_LANG}"
        seen = set()
        count = 0
        new_count = 0
        with open(src_path) as f:
            for line in f:
                count += 1
        process_bar = tqdm(total=count)
        with open(src_path, 'r') as src_fin, open(tgt_path, 'r') as tgt_fin, open(src_output_path, 'w') as src_fout, open(
                tgt_output_path, 'w') as tgt_fout:
            for src_line, tgt_line in zip(src_fin, tgt_fin):
                fil = re.compile(u'[\u4e00-\u9fa5]+', re.UNICODE)
                if re.search(fil, src_line):
                    continue
                h = hash(src_line + " <##> " + tgt_line)
                if h not in seen:
                    src_fout.write(src_line)
                    tgt_fout.write(tgt_line)
                    seen.add(h)
                    new_count += 1
                process_bar.update(1)
            print(f"{count}->{new_count}")
    else:
        os.system(f"mv {data_dir}/{SPLIT}.norm.tok.{SRC_LANG} {data_dir}/{SPLIT}.norm.tok.dedup.{SRC_LANG}")
        os.system(f"mv {data_dir}/{SPLIT}.norm.tok.{TGT_LANG} {data_dir}/{SPLIT}.norm.tok.dedup.{TGT_LANG}")

    for LANG in [SRC_LANG, TGT_LANG]:
        # Truecase
        if LANG != "zh":
            print("开始处理字符：Truecase")
            if SPLIT == "train":
                os.system(
                    f"{TRAIN_TC} --model {model_dir}/truecase-model.{LANG} --corpus {data_dir}/{SPLIT}.norm.tok.dedup.{LANG}")
            os.system(
                f"{TC} --model {model_dir}/truecase-model.{LANG} < {data_dir}/{SPLIT}.norm.tok.dedup.{LANG} > {data_dir}/{SPLIT}.norm.tok.dedup.true.{LANG}")
            os.system(f"mv {data_dir}/{SPLIT}.norm.tok.dedup.true.{LANG} {data_dir}/{SPLIT}.norm.tok.dedup.{LANG}")

    # BPE分词
    if SPLIT == "train":
        os.system(
            f"python {BPEROOT}/learn_joint_bpe_and_vocab.py --input {data_dir}/{SPLIT}.norm.tok.dedup.{SRC_LANG} {data_dir}/{SPLIT}.norm.tok.dedup.{TGT_LANG} -s 30000 -o {model_dir}/bpecode.{SRC_LANG}-{TGT_LANG}.joint --write-vocabulary {model_dir}/voc.{SRC_LANG} {model_dir}/voc.{TGT_LANG}")
    os.system(
        f"python {BPEROOT}/apply_bpe.py -c {model_dir}/bpecode.{SRC_LANG}-{TGT_LANG}.joint < {data_dir}/{SPLIT}.norm.tok.dedup.{SRC_LANG} > {data_dir}/{SPLIT}.{SRC_LANG}-{TGT_LANG}.{SRC_LANG}")
    os.system(
        f"python {BPEROOT}/apply_bpe.py -c {model_dir}/bpecode.{SRC_LANG}-{TGT_LANG}.joint < {data_dir}/{SPLIT}.norm.tok.dedup.{TGT_LANG} > {data_dir}/{SPLIT}.{SRC_LANG}-{TGT_LANG}.{TGT_LANG}")
