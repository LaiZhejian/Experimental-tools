import html
import re
from collections import defaultdict

import jieba

file1 = 'sentd.hter'
file2 = 'sents.hter'


def parse_sag():
    with open(file1) as f1, open(file2) as f2, open("dev.zh-en.en", "w") as f3, open("dev.zh-en.zh", "w") as f4:
        pattern = re.compile(r'<seg id="(\d+)">(.*)</seg>')
        for l1, l2 in zip(f1, f2):
            r1 = re.search(pattern, l1)
            if r1:
                r1 = r1.groups()
            r2 = re.search(pattern, l2)
            if r2:
                r2 = r2.groups()
            if r1 and r2 and r1[0] == r2[0]:
                f3.write(r1[1] + '\n')
                f4.write(" ".join(jieba.cut(r2[1])) + '\n')
                f4.write(r2[1] + '\n')


def parse_xml():
    import xml.etree.ElementTree as ET
    tree = ET.parse('/Users/dream/Downloads/wmttest2022.zh-en.xml')
    root = tree.getroot()
    with open('/Users/dream/Downloads/test.zh-en.zh', 'w') as f1, open('/Users/dream/Downloads/test.zh-en.en',
                                                                       'w') as f2:
        for collection in root:
            for doc in collection:
                src = doc[0][0]
                print(doc.attrib['id'])
                ref = doc[1][0]
                for src_l, ref_l in zip(src, ref):
                    if src_l.attrib['id'] != ref_l.attrib['id']:
                        print("Wrong")
                        return
                    f1.write(" ".join(jieba.cut(html.unescape(src_l.text))) + '\n')
                    f2.write(html.unescape(ref_l.text) + '\n')


def parse_tsv():
    import pandas as pd
    with open('test.zh-en.zh', 'w') as src_f, open('test.zh-en.en', 'w') as tgt_f, \
            open('test.zh-en.tags', 'w') as tag_f:
        df = pd.read_csv('/Users/dream/Downloads/mqm_generalMT2022_zhen.tsv', encoding='utf-8', sep='\t',
                         error_bad_lines=False)
        data = []
        for index, row in df.iterrows():
            if row['severity'] == 'major' or row['severity'] == 'minor':
                tgt = " ".join(row['target'].split())
                result = re.search('<v>(.*?)</v>', tgt)
                src = row['source']
                if len(tgt.split()) > 1024 or len(src.split()) > 1024: continue
                if result and result.groups()[0] != ' ':
                    tgt = re.sub('<v>(.*?)</v>', " " + result.groups()[0] + " ", tgt)
                    tgt = " ".join(tgt.split())
                    data.append((result.span()[0], result.span()[1] - 7, src, tgt, row['severity']))
                    # print(tgt[result.span()[0]: result.span()[1] - 7])
                # else:
                #     result = re.search('<v>(.*?)</v>', src)
                #     src = re.sub('<v>(.*?)</v>', result.groups()[0], src)
                #     if result is None:
                #         print(index)
                #     data[(src, tgt)].append((len(tgt), len(tgt) + 1, row['severity']))
        for left, right, src, tgt, severity in data:
            src_f.write(" ".join(jieba.cut(src)) + '\n')
            tgt_f.write(tgt + '\n')
            bound = ""
            bound += len(tgt[:left].split()) * "OK "
            bound += len(tgt[left:right].split()) * (severity + " ")
            # print(tgt[item[0]:item[1]])
            bound += "OK " * (len(tgt.split()) - len(bound.split()))
            tag_f.write(bound + '\n')
            # if len(result.groups()) != 0:
            #     print(index)

    # print(df)


parse_tsv()
