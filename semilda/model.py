#coding: utf-8

class Topic:

    def __init__(self, topic_name):
        self.topic_name = topic_name
        self.word_list = []

class Rule:

    def __init__(self):
        pass


class Model:
    
    def __init__(self):
        self.topic_list = []

    def init_model(self):
        pass

    def loglikelihood(self):
        pass

    def stat_corpus(self):
        pass


class Sampler:

    def init(self, model, corpus):
        self.model = model
        self.corpus = corpus

    def sample_corpus(self):
        pass

    def sample_doc(self):
        pass

    def sample_word(self):
        pass
