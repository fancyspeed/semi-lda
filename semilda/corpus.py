#coding: utf-8

class Corpus:

    def __init__(self, args):
    self.args = args
    self.doc_list = []

    def init_corpus(self): 
        pass


class Document:
    
    def __init__(self):
        self.word_list = []

    def add_word(self, word):
        self.word_list.append( word )


class Word:
    
    def __init__(self, word):
        self.word = word
        self.topic = None

    def assign(self, topic):
        self.topic = topic

