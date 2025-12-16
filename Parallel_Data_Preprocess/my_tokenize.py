"""
针对不同语言的tokenize程序
"""
from tqdm import tqdm
import argparse
import pkuseg
import subprocess
import sacremoses
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='input file')
    parser.add_argument('-l', type=str, required=True)
    parser.add_argument('--no-escape', default=False, action="store_true",
                        help='don\'t perform HTML escaping on apostrophy, quotes, etc.')
    parser.add_argument('--output-file', type=str,
                        default=None, help='output file')
    args = parser.parse_args()

    # 处理中文分词
    if args.l == 'zh':
        seg = pkuseg.pkuseg()
    tokenizer = sacremoses.MosesTokenizer(lang=args.l)

    # 获取行数，方便获得进度条
    result = subprocess.run(['wc', '-l', args.INPUT], stdout=subprocess.PIPE)
    line_number = int(result.stdout.decode('utf-8').split()[0])

    # 处理无输出文件的情况
    if args.output_file is None:
        output_file = sys.stdout
    else:
        output_file = open(args.output_file, 'w')

    # 开始处理数据
    with open(args.INPUT, 'r') as input_file:
        for line in tqdm(input_file, total=line_number):
            line = line.strip()
            if args.l == "zh":
                line = " ".join(seg.cut(line))
            output_file.write(tokenizer.tokenize(
                line, escape=not args.no_escape, return_str=True) + "\n")

    output_file.close()
