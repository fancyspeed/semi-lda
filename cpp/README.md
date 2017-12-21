# input for documents

* allow label or not for each document. If no labels, leave a space in the begining
* such as,

label1 word1 word1 word2 word2 word3
label1,label2 word1 word1 word2 word2 word3
 word1 word1 word2 word2 word3
 word5 word6 word6 ...

# input for seed words

* for each line, need topic id, topic name, and a set of seed words
* do not need seed words for every topics. If no seed words at all, it's unsupervised
* such as,

topicid1 topicname1 word1 word2 word3 ...
topicid2 topicname2 word1 word2 ...

# run

* prepare: convert raw data to PLDA format and extract word index file

python trans_train.py ../data/train_post_labeltag ../data/train_model_input ../data/train_topicmap ../data/train_wordindex

* train: given number of topics, alpha, beta, input file, output model file. seed file is optional

sh -x lda_semi_train.sh
  or 
sh -x lda_train.sh

