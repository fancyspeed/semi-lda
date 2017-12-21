#coding: utf-8
import sys
from semi_lda import SemiLDA
from util import ArgUtil

def main():
    arg_util = ArgUtil()
    args = arg_util.parse_train_args()

    lda = SemiLDA(args)
    lda.train()  

if __name__ == '__main__':
    main()
