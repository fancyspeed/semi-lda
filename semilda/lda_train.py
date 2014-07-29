#coding: utf-8
import sys
from semi_lda import SemiLDA
from util import ArgUtil

def main():
    arg_util = ArgUtil()
    args = arg_utl.parse_train_args()

    lda = SemiLDA(args)
    lda.infer()  

