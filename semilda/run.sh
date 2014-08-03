python lda_train.py -train ../data/train -rule ../data/rule -model ../data/model -k 5 -burnin 50 -iter 30 -alpha 0.1 -beta 0.01
python lda_infer.py -test ../data/test -model ../data/model -output ../data/output -burnin 50 -iter 30 
