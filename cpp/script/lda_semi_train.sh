#!/bin/sh -x
ldapath=../src
train_file=../data/train_model_input
seed_file=../data/seed_words
index_file=../data/train_wordindex
model_file=../data/model
view_file=../data/viewable_model

num_topic=100
#`wc -l ../data/seed_words | awk -F" " '{print $1}'`
alpha=0.5
beta=0.01

time mpiexec -n 4 $ldapath/mpi_slda \
--num_topics $num_topic \
--alpha $alpha --beta $beta \
--training_data_file $train_file \
--model_file $model_file \
--word_index_file $index_file \
--compute_likelihood true \
--seed_word_file $seed_file \
--burn_in_iterations 50 --total_iterations 100

python gen_viewable_model.py $model_file $view_file

