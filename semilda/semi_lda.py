#coding: utf-8
from corpus import Corpus
from model import Model
from sampler import Sampler

class SemiLDA(object):
    
    def __init__(self, args):
        self.args = args
        self.model = Model()

        try: # test: load model
            if args.test:
                self.model.load_model(args.model)    
        except: # train: init corpus and model
            self.corpus = Corpus()
            self.model.init_model(args)
            if args.rule:
                self.model.load_rules(args.rule)
            self.corpus.init_corpus_and_model(args.train, self.model) 
        # init sampler
        self.sampler = Sampler(self.model)

    def train(self):
        for i in range(self.args.burn_in):
            self.sampler.sample_corpus(self.corpus)
            if not self.args.slient:
                self.sampler.loglikelihood(self.corpus)
        for i in range(self.args.max_iter):
            self.sampler.sample_corpus(self.corpus)
            self.model.accumulative()
            if not self.args.slient:
                self.sampler.loglikelihood(self.corpus)
        self.model.save_model(self.args.model)

    def infer(self):
        self.sampler.sample_test(self.args.test, self.args.output, self.args.burn_in, self.args.max_iter)


def test():
    pass

if __name__ == '__main__':
    test()
