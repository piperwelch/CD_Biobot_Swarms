#!/bin/bash

# generates pickle files of formatted results

BOT_ID=$1

BASE_DIR=$HOME/research/anthrobots/anthrobot-sim

# * is all directories... *res16* will do all res16 folders for the specified bot.. *ctrl2 will do all ctrl2s etc.
for dir in $BASE_DIR/output_data/$BOT_ID/*; do 
    echo "$dir"
    python3 $BASE_DIR/data_analysis/read_results.py $dir
done
