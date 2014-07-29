#coding: utf-8
import sys
from semi_lda import SemiLDA
from util import ArgUtil

if __name__ == '__main__':
    util = ArgUtil()
    args = util.parse_train_args()

    lda = SemiLDA(args)
    lda.train()  

