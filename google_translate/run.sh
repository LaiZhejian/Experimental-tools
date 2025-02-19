#!/bin/bash

# pip install -r requrements.txt

input_file=$1
target_lang=$2

python translate.py \
    --sleep 0.5 \
    --source_lang auto \
    --target_lang $target_lang \
    --input_file $input_file \
    --output_file $input_file.google_translate.$target_lang \
    --resume 
