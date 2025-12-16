import os
import subprocess
import numpy as np
import pandas as pd
import sacremoses
from langconv import *
import html
import re
import string
import pkuseg
from tqdm.contrib.concurrent import process_map
import click

seg = pkuseg.pkuseg()
tradition2Simplifiedconverter = Converter('zh-hans')
SRC_LANG = None
TGT_LANG = None
mpn = None
mtn = None
skip = False
repeat_regex = r'(?:(.)\1{4,}|(..)\2{3,}|(...)\3{2,})'

char_ranges = {}
char_ranges['de'] = [
    (0x41, 0x5A),    # 大写字母 A-Z
    (0x61, 0x7A),    # 小写字母 a-z
    (0xC4, 0xC4), (0xD6, 0xD6), (0xDC, 0xDC),  # 大写 Ä, Ö, Ü
    (0xE4, 0xE4), (0xF6, 0xF6), (0xFC, 0xFC),  # 小写 ä, ö, ü
    (0xDF, 0xDF)              # Eszett ß
]

char_ranges['en'] = [
    (0x41, 0x5A),    # 大写字母 A-Z
    (0x61, 0x7A),    # 小写字母 a-z
    # (0x30, 0x39),    # 数字 0-9
    # (0x21, 0x2F),    # 标点符号
    # (0x3A, 0x40),    # 标点符号
    # (0x5B, 0x60),    # 标点符号
    # (0x7B, 0x7E)     # 标点符号
]

char_ranges['fr'] = [
    (0x0041, 0x005A),   # 大写字母
    (0x0061, 0x007A),   # 小写字母
    (0x00C0, 0x00FF)    # 重音字母等
]

char_ranges['he'] = [
    (0x0590, 0x05FF)
]

char_ranges['ja'] = [
    (0x4E00, 0x9FFF),   # 基本汉字共通部分
    (0x3040, 0x309F),
    (0x30A0, 0x30FF)
]

char_ranges['ko'] = [
    (0xAC00, 0xD7AF),   # 韩文音节
    (0x1100, 0x11FF)    # 韩文辅助字母
]

char_ranges['ru'] = [
    (0x0400, 0x04FF)
]

char_ranges['zh'] = [
    (0x4E00, 0x9FFF),  # 基本汉字
    (0x3400, 0x4DBF),  # 扩展 A
    (0x20000, 0x2A6DF),  # 扩展 B
    (0x2A700, 0x2B73F),  # 扩展 C
    (0x2B740, 0x2B81F),  # 扩展 D
    (0x2B820, 0x2CEAF),  # 扩展 E
    (0xF900, 0xFAFF),  # 兼容汉字
    (0x2F800, 0x2FA1F),  # 兼容扩展
]


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


def count_native_characters(chr_text, lang):

    def is_chinese_char(c):
        return any(start <= ord(c) <= end for start, end in char_ranges[lang])

    return sum(1 for char in chr_text if is_chinese_char(char))


def process(args):
    global repeat_regex
    src_line, tgt_line = args

    # 1. Remove empty sentences
    src_line, tgt_line = src_line.strip(), tgt_line.strip()
    if src_line == "" or tgt_line == "":
        return " ".join(src_line), " ".join(tgt_line) if not_skip else None
    # 2. Eliminate escaped HTML characters
    src_line, tgt_line = html.unescape(src_line), html.unescape(tgt_line)
    # 3. Normalize spacing and punctuation
    src_line, tgt_line = mpn.normalize(src_line), mpn.normalize(tgt_line)
    # 4. Remove sentences with repetition marks
    if re.search(repeat_regex, src_line) or re.search(repeat_regex, tgt_line):
        return " ".join(src_line), " ".join(tgt_line) if not_skip else None
    # 4.5 中文分词(if need)
    if SRC_LANG == 'zh':
        src_line = " ".join(seg.cut(src_line))
        src_line = full_to_half(src_line)
    if TGT_LANG == 'zh':
        tgt_line = " ".join(seg.cut(tgt_line))
        tgt_line = full_to_half(tgt_line)
    # 5. tokenization
    src_line = mt[SRC_LANG].tokenize(src_line, escape=False)
    tgt_line = mt[TGT_LANG].tokenize(tgt_line, escape=False)
    if not_skip:
        return " ".join(src_line), " ".join(tgt_line)
    # 6. Delete sentence pairs with inconsistent punctuation at the end of the original and translated texts.
    if src_line[-1] in string.punctuation and tgt_line[-1] in string.punctuation and src_line[-1] != tgt_line[-1]:
        return None
    # 7. Remove sentence pairs with a source/target token ratio exceeding 1:3 (or 3:1).
    if len(src_line) > 3 * len(tgt_line) or len(tgt_line) > 3 * len(src_line):
        return None
    # 8. Delete segments that exceed 150 tokens in length.
    if len(src_line) > 150 or len(tgt_line) > 150:
        return None
    # 9. Remove sentence pairs with fewer than 5 tokens in the source text or translation.
    if len(src_line) < 5 or len(tgt_line) < 5:
        return None
    src_line = " ".join(src_line)
    tgt_line = " ".join(tgt_line)
    # 9.5 Convert traditional Chinese characters to simplified Chinese characters(if needed)
    if SRC_LANG == 'zh':
        src_line = tradition2Simplifiedconverter.convert(src_line)
    if TGT_LANG == 'zh':
        tgt_line = tradition2Simplifiedconverter.convert(tgt_line)
    # 10. Delete corpora with an unaligned number of parentheses
    if src_line.count('(') != tgt_line.count('(') or src_line.count(')') != tgt_line.count(')'):
        return None
    # 11. Delete corpora with an unaligned number of Arabic numerals
    if len(re.findall(r'\d+', src_line)) != len(re.findall(r'\d+', tgt_line)):
        return None
    # 12. Remove corpora with non-native character ratios greater than 0.4.
    chr_src_line = src_line.replace(" ", "")
    chr_tgt_line = tgt_line.replace(" ", "")
    if count_native_characters(chr_src_line, SRC_LANG) < 0.4 * len(chr_src_line) or \
            count_native_characters(chr_tgt_line, TGT_LANG) < 0.4 * len(chr_tgt_line):
        return None

    return src_line, tgt_line


@click.command()
@click.option('--src-path', required=True, type=str)
@click.option('--tgt-path', required=True, type=str)
@click.option('--output-dir', required=True, type=str)
@click.option('--src-lang', required=True, type=str)
@click.option('--tgt-lang', required=True, type=str)
@click.option('--subset', required=True, type=str)
@click.option('--skip', default=False, is_flag=True, type=bool)
def main(src_path, tgt_path, output_dir, src_lang, tgt_lang, subset, skip):
    global SRC_LANG
    global TGT_LANG
    global mpn
    global mt
    global not_skip

    not_skip = not skip

    SRC_LANG = src_lang
    TGT_LANG = tgt_lang

    mpn = sacremoses.MosesPunctNormalizer()
    mt = {SRC_LANG: sacremoses.MosesTokenizer(
        lang=SRC_LANG), TGT_LANG: sacremoses.MosesTokenizer(lang=TGT_LANG)}

    try:
        result = subprocess.run(['wc', '-l', src_path], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, check=True)
        # 获取命令的标准输出并打印行数
        output = result.stdout
        line_count = int(output.split()[0])
        print(f'文件"{src_path}"的行数为: {line_count}')
    except subprocess.CalledProcessError as e:
        print(f'执行命令时出错: {e}')

    answers = None

    with open(src_path) as src_file, open(tgt_path) as tgt_file:

        src_file = src_file.readlines()
        tgt_file = tgt_file.readlines()

        answers = process_map(process, list(
            zip(src_file, tgt_file)), total=line_count, max_workers=32, chunksize=100)

        answers = [ans for ans in answers if ans is not None]

    print("Total number of sentence pairs: ", len(answers))

    with open(output_dir + subset + "." + SRC_LANG, "w") as src_file, open(output_dir + subset + "." + TGT_LANG, "w") as tgt_file:
        for src_line, tgt_line in answers:
            src_file.write(src_line + "\n")
            tgt_file.write(tgt_line + "\n")


if __name__ == "__main__":
    main()
