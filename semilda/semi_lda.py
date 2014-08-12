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
        for i in xrange(self.args.burn_in):
            self.sampler.sample_corpus(self.corpus)
            if not self.args.slient:
                loglike = self.sampler.loglikelihood(self.corpus)
                print 'burn in:%s, loglikelihood:%s' % (i, loglike) 
        for i in xrange(self.args.max_iter):
            self.sampler.sample_corpus(self.corpus)
            self.model.accumulative()
            if not self.args.slient:
                loglike = self.sampler.loglikelihood(self.corpus)
                print 'iter:%s, loglikelihood:%s' % (i, loglike) 
        self.model.save_model(self.args.model)
        if self.args.dump:
            self.model.dump_topic_words(self.args.dump)

    def infer(self):
        self.sampler.sample_test(self.args.test, self.args.output, self.args.burn_in, self.args.max_iter)


