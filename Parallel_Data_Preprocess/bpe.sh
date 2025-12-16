BPE_ROOT=/home/nfs03/laizj/code/util/subword-nmt/subword_nmt
TOKENIZE_SCRIPT=/home/nfs03/laizj/code/util/Preprocess/my_tokenize.py

SRC_LANG=zh
TGT_LANG=en
input_dir=/home/nfs02/laizj/experiment/WMT2023/sescore
input_file_prefix=test
output_dir=/home/nfs02/laizj/experiment/WMT2023/sescore/bin
SPLIT=bpe # outputdir/SPLIT.SRC_LANG-TGT_LANG
codes_file=/home/nfs03/laizj/backup/WMT2023/generator/zhen-nmt-fairseq/total/bpecode.zh-en.joint

mkdir -p ${output_dir}
cp ${BASH_SOURCE[0]} $output_dir/$input_file_prefix.bpe.sh

python ${TOKENIZE_SCRIPT} ${input_dir}/${input_file_prefix}.${SRC_LANG} -l ${SRC_LANG} --output-file ${output_dir}/${SPLIT}.tok.${SRC_LANG}-${TGT_LANG}.${SRC_LANG} --no-escape
python ${TOKENIZE_SCRIPT} ${input_dir}/${input_file_prefix}.${TGT_LANG} -l ${TGT_LANG} --output-file ${output_dir}/${SPLIT}.tok.${SRC_LANG}-${TGT_LANG}.${TGT_LANG} --no-escape

# python ${BPE_ROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/${SPLIT}.${SRC_LANG}-${TGT_LANG}.${SRC_LANG}.final ${data_dir}/${SPLIT}.${SRC_LANG}-${TGT_LANG}.${TGT_LANG}.final -s 30000 -o ${model_dir}/bpecode.${SRC_LANG}-${TGT_LANG}.joint --write-vocabulary ${data_dir}/${SRC_LANG}-${TGT_LANG}.${SRC_LANG}.vocab ${data_dir}/${SRC_LANG}-${TGT_LANG}.${TGT_LANG}.vocab
python ${BPE_ROOT}/apply_bpe.py -c ${codes_file} < ${output_dir}/${SPLIT}.tok.${SRC_LANG}-${TGT_LANG}.${SRC_LANG} > ${output_dir}/${SPLIT}.${SRC_LANG}-${TGT_LANG}.${SRC_LANG}
python ${BPE_ROOT}/apply_bpe.py -c ${codes_file} < ${output_dir}/${SPLIT}.tok.${SRC_LANG}-${TGT_LANG}.${TGT_LANG} > ${output_dir}/${SPLIT}.${SRC_LANG}-${TGT_LANG}.${TGT_LANG}

#fairseq-preprocess \
# --source-lang ${SRC_LANG} \
# --target-lang ${TGT_LANG} \
# --trainpref ${data_dir}/train.${SRC_LANG}-${TGT_LANG}.bpe \
# --validpref ${data_dir}/dev.${SRC_LANG}-${TGT_LANG}.bpe \
# --testpref ${data_dir}/test.${SRC_LANG}-${TGT_LANG}.bpe \
# --destdir ${data_dir}/bin \
# --thresholdtgt 0 \
# --thresholdsrc 0 \
# --srcdict ${data_dir}/${SRC_LANG}-${TGT_LANG}.${SRC_LANG}.vocab \
# --tgtdict ${data_dir}/${SRC_LANG}-${TGT_LANG}.${TGT_LANG}.vocab \
# --workers 70