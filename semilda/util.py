#!/usr/bin/env python

import sys
#import getopt
import argparse

class ArgUtil(object):

    def __init__(self):
        self.base_parser = argparse.ArgumentParser(add_help=False)
        self.base_parser.add_argument('-model', help='model file path')
        self.base_parser.add_argument('-alpha', help='alpha')
        self.base_parser.add_argument('-beta', help='beta')
        self.base_parser.add_argument('-burnin', dest='burn_in', type=int, default=20, help='burn in iteration')

        self.train_parser = argparse.ArgumentParser(parents=[self.base_parser],
                                         description='Semi-supervised LDA.',
                                         epilog='if inference, please use lda_infer.py')
        self.train_parser.add_argument('-train', help='training data')
        self.train_parser.add_argument('-rule', nargs='?', help='rule data (optional)')
        self.train_parser.add_argument('-k', help='total number of topics')
        self.train_parser.add_argument('-iter', dest='max_iter', type=int, default=50, help='total iteration')

        self.infer_parser = argparse.ArgumentParser(parents=[self.base_parser],
                                         description='Semi-supervised LDA.',
                                         epilog='if training, please use lda_train.py')
        self.infer_parser.add_argument('-test', help='testing data')
        self.infer_parser.add_argument('-output', help='output path of inference')

    def parse_train_args(self):
        args = self.train_parser.parse_args()
        print args.train

    def parse_infer_args(self):
        args = self.infer_parser.parse_args()
        print args.test

if __name__ == '__main__':
    util = ArgUtil()
    util.parse_train_args()


