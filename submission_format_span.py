import os
import re
import numpy as np

path = "/home/nfs01/gengx/submit/word/en-de/10"

mt_tok = "/home/data_91_c/laizj/data/WMT/23/mqm/real/en-de/test/test.en-de.de"
mt_raw = "/home/data_91_c/laizj/data/WMT/23/mqm/real/en-de/test/test.en-de.raw.de"

indices = {}
prob_file = open(path+"/test.prob","r",encoding="utf-8")
prob_lines = prob_file.readlines()
mt_tok_file = open(mt_tok, "r", encoding="utf-8")
mt_tok_lines = mt_tok_file.readlines()
mt_raw_file = open(mt_raw, "r", encoding="utf-8")
mt_raw_lines = mt_raw_file.readlines()
major_cnt = 0
minor_cnt = 0
for sid, zip_line in enumerate(zip(prob_lines, mt_tok_lines, mt_raw_lines)):
    prob_line = zip_line[0]
    mt_tok_line = zip_line[1]
    mt_raw_line = zip_line[2]
    mt_raw_str = mt_raw_line.strip()
    prob_line = prob_line.split()
    mt_tok_line = mt_tok_line.split()
    mt_raw_line = mt_raw_line.split()
    mt_raw_probs = []
    tmp_probs = []
    for mt_tok, prob in zip(mt_tok_line, prob_line):
        for i in range(len(mt_raw_line)):
            if mt_tok == "<EOS>":
                continue
            if mt_tok in mt_raw_line[i]:
                mt_raw_line[i] = mt_raw_line[i].replace(mt_tok, "",1)
                tmp_probs.append(float(prob))
                if mt_raw_line[i] == "":
                    mt_raw_probs.append(tmp_probs)
                    tmp_probs = []
                break
    for i in range(len(mt_raw_line)):
        mt_raw_probs[i] = min(mt_raw_probs[i])
    # print(mt_raw_probs)
    assert len(mt_raw_line) == len(mt_raw_probs)

    mt_raw_line = mt_raw_str.split()
    indices[sid] = []
    start = -1
    idx = 0
    error = "no-error"
    cnt_major = 0
    cnt_minor = 0
    for i in range(len(mt_raw_line)):
        if mt_raw_probs[i] < 0.905:  #major
            if start == -1:
                start = idx
                error = "major"
            else:
                # 当span出现major时就视作major
                error = "major"
            cnt_major = cnt_major + 1
        elif mt_raw_probs[i] < 0.91:
            if start == -1:
                start = idx
                error = "minor"
            else:
                pass
            cnt_minor = cnt_minor + 1
        else:
            if start == -1:
                pass
            else:
                # if cnt_major > 0:
                #     error = "major"
                if cnt_major >= cnt_minor:
                    error = "major"
                else:
                    error = "minor"
                indices[sid].append([start,idx-1,error])
                start = -1
                if error == "major":
                    major_cnt = major_cnt + 1
                elif error == "minor":
                    minor_cnt = minor_cnt + 1
                cnt_major = 0
                cnt_minor = 0
        idx = idx + len(mt_raw_line[i])+1 # 字符长度+空格
    if start == -1:
        pass
    else:
        # if cnt_major > 0:
        #     error = "major"
        if cnt_major >= cnt_minor:
            error = "major"
        else:
            error = "minor"
        indices[sid].append([start, idx - 1, error])
        start = -1
        if error == "major":
            major_cnt = major_cnt + 1
        elif error == "minor":
            minor_cnt = minor_cnt + 1
    if not indices[sid]:
        indices[sid].append([-1, - 1, error])


res=path+"/predictions.txt"
with open(res, "w", encoding="utf-8") as w:
    w.write("3264730349\n")
    w.write("560145557\n")
    w.write("12\n")
    for sid, mt_raw_line in enumerate(mt_raw_lines):
        mt_raw_str = mt_raw_line.strip()
        spans = indices[sid]
        starts = []
        ends = []
        sevs = []
        for sp in spans:
            starts.append(str(sp[0]))
            ends.append(str(sp[1]))
            sevs.append(str(sp[2]))
        starts, ends, sevs = zip(*sorted(zip(starts, ends, sevs)))
        start = " ".join(starts)
        end = " ".join(ends)
        sev = " ".join(sevs)
        w.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n".format("en-de", "njuqe_10", sid, mt_raw_str, start, end, sev))

print(major_cnt)
print(minor_cnt)