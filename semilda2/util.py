#!/usr/bin/env python

import sys
#import getopt
import argparse

class ArgUtil(object):

    def __init__(self):
        self.base_parser = argparse.ArgumentParser(add_help=False)
        self.base_parser.add_argument('-v', '--version', action='version', version='v0.1')
        self.base_parser.add_argument('-model', required=True, help='model file path')
        self.base_parser.add_argument('-burnin', dest='burn_in', type=int, default=20, help='burn in iteration')
        self.base_parser.add_argument('-iter', dest='max_iter', type=int, default=50, help='total iteration')

        self.train_parser = argparse.ArgumentParser(parents=[self.base_parser],
                                         description='Semi-supervised LDA.',
                                         epilog='if inference, please use lda_infer.py')
        self.train_parser.add_argument('-train', required=True, help='training data')
        self.train_parser.add_argument('-rule', nargs='?', help='rule data (optional)')
        self.train_parser.add_argument('-dump', nargs='?', help='dump topic words (optional)')
        self.train_parser.add_argument('-alpha', type=float, default=0.5, help='alpha')
        self.train_parser.add_argument('-beta', type=float, default=0.1, help='beta')
        self.train_parser.add_argument('-k', dest='num_topic', type=int, required=True, help='total number of topics')
        self.train_parser.add_argument('-s', '--slient', action='store_true', help='no likelihood')

        self.infer_parser = argparse.ArgumentParser(parents=[self.base_parser],
                                         description='Semi-supervised LDA.',
                                         epilog='if training, please use lda_train.py')
        self.infer_parser.add_argument('-test', required=True, help='testing data')
        self.infer_parser.add_argument('-output', required=True, help='output path of inference')

    def parse_train_args(self):
        args = self.train_parser.parse_args()
        return args

    def parse_infer_args(self):
        args = self.infer_parser.parse_args()
        return args

class DataLoader(object):
    
    @staticmethod
    def load_train(p_train):
        #topic_dict = {}
        doc_list = []

        f=lambda v: tuple(v.split(':', 1)) if v.count(':')>=1 else (v, 1)

        for line in open(p_train):
            row = line.rstrip().split(' ')
            label_list = [v for v in row[0] if v]
            word_list = [f(v) for v in row[1:] if v]
            doc_list.append( (word_list, label_list) )
        return doc_list

    @staticmethod
    def load_rule(p_rule):
        pass

    @staticmethod
    def load_test(p_test):
        pass

if __name__ == '__main__':
    util = ArgUtil()
    print util.parse_train_args()

    train = DataLoader.load_train(sys.argv[2])

