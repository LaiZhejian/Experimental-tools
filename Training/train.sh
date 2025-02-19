TRAIN=train
VALID=valid
SRC=en
MT=de
NJUQE=/home/nfs01/gengx/code/NJUQE-NJUAPE
save_dir=/home/laizj/pretrain_ssqe
mkdir -p $save_dir

export CUDA_VISIBLE_DEVICES=3
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

# fairseq-preprocess \
# --source-lang $SRC  \
# --target-lang $MT \
# --srcdict  $BIN/dict.$SRC.txt  \
# --tgtdict  $BIN/dict.$MT.txt  \
# --trainpref  $BIN/$TRAIN.$SRC-$MT \
# --validpref  $BIN/$VALID.$SRC-$MT \
# --destdir $BIN \
# --dataset-impl mmap \
# --workers  60

# # 转化为fairseq的二进制tags
# fairseq-preprocess \
# --source-lang tag  \
# --srcdict  $BIN/dict.tag.txt  \
# --trainpref  $BIN/../train.tag-None \
# --validpref  $BIN/../valid.tag-None \
# --destdir $BIN \
# --only-source \
# --dataset-impl mmap \
# --workers  60

python $NJUQE/fairseq_cli/train.py \
    /home/data_91_c/laizj/data/WMT/20/en-zh/sentencepiece \
    --restore-file /home/data_ti6_c/zhangy/model/xlmr.large/model.pt\
    -s en -t zh --word-score-suffix tags --sent--score-suffix hter --bounds mt \
    --mask-prob 0.15 --mask-whole-words \
    --dev-N 7 --dev-M 1 --test-N 40 --test-M 6\
    --arch roberta_large --task self_supervised_qe_task --criterion ssqe_loss \
    --bpe sentencepiece \
    --optimizer adam --adam-eps 1e-08 --adam-betas '(0.9, 0.99)' \
    --dataset-impl raw \
    --reset-meters --reset-optimizer --reset-dataloader --reset-lr-scheduler\
    --lr 5e-6 --lr-scheduler polynomial_decay --warmup-updates 10000 \
    --patience 25 --validate-interval-updates 1000 \
    --reset-meters --reset-optimizer --reset-dataloader \
    --no-epoch-checkpoints --no-last-checkpoints --keep-last-epochs 1 \
    --best-checkpoint-metric pearson --maximize-best-checkpoint-metric \
    --update-freq 2 --batch-size 2 --batch-size-valid 1 --seed 20 \
    --user-dir ${NJUQE}/njuqe \
    --save-dir ${save_dir} \
    --fine-tune \
    --tensorboard-logdir ${save_dir}/logdir > ${save_dir}/train.log 

# bash /home/nfs01/gengx/code/NJUQE-NJUAPE/experiments/wmt2023/pretrain_xlmr_ende_8141.sh > /home/gengx/pretrain_xlmr_ende_8141.log 2>&1 &