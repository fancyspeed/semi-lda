#!/bin/sh -x

test_file=../data/test_raw
python trans_test.py $test_file ../data/test

ldapath=../src
test_path=../data/test
pred_path=../data/pred
model_file=../data/smodel

args="--alpha 0.5 \
      --beta 0.01 \
      --inference_data_file ${test_path} \
      --inference_result_file ${pred_path} \
      --model_file ${model_file} \
      --burn_in_iterations 50 \
      --total_iterations 100 \
      --file_type 0
      "

time $ldapath/infer $args

result_path=../../../user_modeling/interest/out/lda.pred
label_map=../data/label_map

python trans_pred.py $label_map $pred_path $result_path

