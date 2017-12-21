#!/bin/sh -x

python pre_trans.py ../data/train_raw_labeltag ../data/train_post_labeltag

python trans_train.py ../data/train_post_labeltag ../data/train_model_input ../data/train_topicmap ../data/train_wordindex
