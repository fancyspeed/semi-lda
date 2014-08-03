
## Description

Semi-supervised LDA, which allows labelling of some documents in training, and a rule file to assign seed words for each topic.

Besides topics appear in training data and the rule file, new topics can also be learned.

## Data format

* training data:

    label1,label2 word1:num1 word2:num2 ...

    word3:num3 word4:num4 ...

* rule file:

    label1 word1,word2 ...

    label2 word3,word4 ...

(also compatible with LibSVM.)

## Example

* train:

    $ python lda\_train.py -train train.txt -rule rule.txt -k 5 -alpha 0.1 -beta 0.01 -burnin 50 -iter 50 -model model.txt

* inference:

    $ python lda\_infer.py -model model.txt -test test.txt -output output.txt -burnin 50 -iter 50

## Evaluation

Model performance is evaluated by `log-likelihood`.

## Other plans

* downgrade hot words
* support continuous-valued word frequences
* utilization of contextual information
