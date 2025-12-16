import sentencepiece as spm
import sys

def recover_data(file_path, output_path):

    sp = spm.SentencePieceProcessor()
    sp.load('/home/data_91_c/laizj/code/util/sentencepiece/sentencepiece.bpe.model')

    token_res = []
    with open(file_path) as f:
        for line in f:
            tokens = line.strip().split()
            res = sp.DecodePieces(tokens)
            token_res.append(res)

    with open(output_path, "w") as o:
        for line in token_res:
            o.write(line.strip()+"\n")


if __name__ == '__main__':

    src_path = sys.argv[1]
    out_path = sys.argv[2]

    recover_data(src_path, out_path)


    print("all finished")