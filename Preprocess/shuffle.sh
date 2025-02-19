#!/usr/bin/env bash
SRC=en
MT=de
SEED=40

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}
shuf --random-source=<(get_seeded_random $SEED) /home/data_91_c/laizj/data/WMT/23/parallel/wmt23-zhen/train.zh-en.zh.final | head -n 3000000 > /home/data_91_c/laizj/data/WMT/23/parallel/wmt23-zhen/300w_1/train.zh-en.raw.zh
shuf --random-source=<(get_seeded_random $SEED) /home/data_91_c/laizj/data/WMT/23/parallel/wmt23-zhen/train.zh-en.en.final | head -n 3000000 > /home/data_91_c/laizj/data/WMT/23/parallel/wmt23-zhen/300w_1/train.zh-en.raw.en