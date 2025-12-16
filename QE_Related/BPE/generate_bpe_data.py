from transformers import AutoTokenizer


import sys

def generate_data(file_path, output_path, bounds_path=None):
    tokenizer = AutoTokenizer.from_pretrained('/home/data_91_c/laizj/model/bert-base-multilingual-cased', cache_dir=None, use_fast=False,do_lower_case=False)

    token_res = []
    bounds_res = []
    count = 0
    with open(file_path) as f:
        for line in f:
            count += 1
            token_pieces = ""
            bounds = ""
            offset = 0
            tokens = line.strip().split()
            for token in tokens:
                pieces = tokenizer.tokenize(token)
                for piece in pieces:
                    token_pieces += "".join(piece) + " "
                bounds += str(offset) + " "
                offset += len(pieces)
            token_res.append(token_pieces)
            bounds_res.append(bounds)
            if count % 1000 == 0:
                print(count)

    with open(output_path, "w") as o:
        for line in token_res:
            o.write(line.strip() + "\n")
    if bounds_path:
        with open(bounds_path, "w") as b:
            for line in bounds_res:
                b.write(line.strip() + "\n")


if __name__ == '__main__':

    file_path = "/home/data_91_c/laizj/data/WMT/20/en-zh/dev/"
    out_path = "/home/data_91_c/laizj/data/SelfSupervisedQE/20en-zh/bert_bpe/"
    src = "en"
    mt = "zh"

    train = False
    dev = True
    test = False

    # mt
    if train:
        train_mt = file_path + "train.mt"
        train_mt_output = out_path + f"train.{src}-{mt}.{mt}"
        train_mt_bounds = out_path + f"train.{mt}.bounds"
        generate_data(train_mt, train_mt_output, train_mt_bounds)

    if dev:
        valid_mt = file_path + "dev.mt"
        valid_mt_output = out_path + f"valid.{src}-{mt}.{mt}"
        valid_mt_bounds = out_path + f"valid.{mt}.bounds"
        generate_data(valid_mt, valid_mt_output, valid_mt_bounds)

    if test:
        test_mt = file_path + "test.mt"
        test_mt_output = out_path + f"test.{src}-{mt}.{mt}"
        test_mt_bounds = out_path + f"test.{mt}.bounds"
        generate_data(test_mt, test_mt_output, test_mt_bounds)

    # src
    if train:
        train_src = file_path + "train.src"
        train_src_output = out_path + f"train.{src}-{mt}.{src}"
        train_src_bounds = out_path + f"train.{src}.bounds"
        generate_data(train_src, train_src_output, train_src_bounds)

    if dev:
        valid_src = file_path + "dev.src"
        valid_src_output = out_path + f"valid.{src}-{mt}.{src}"
        valid_src_bounds = out_path + f"valid.{src}.bounds"
        generate_data(valid_src, valid_src_output, valid_src_bounds)

    if test:
        test_src = file_path + "test.src"
        test_src_output = out_path + f"test.{src}-{mt}.{src}"
        test_src_bounds = out_path + f"test.{src}.bounds"
        generate_data(test_src, test_src_output, test_src_bounds)

    print("all finished")
