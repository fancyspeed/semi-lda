#coding: utf-8
import random
import math

class Document:
    
    def __init__(self, word_list=[], label_list=[], topic_count={}):
        self.word_list = word_list 
        self.label_list = label_list 
        self.topic_count = topic_count
        self.accu_topic_count = {}

    def accumulative(self):
        for topic, count in self.topic_count.items():
            self.accu_topic_count[topic] = self.accu_topic_count.get(topic, 0) + count 

class Corpus:

    def __init__(self):
        self.doc_list = []

    @staticmethod
    def init_doc(line, model, update=True):
        if line.rstrip() and line[0]!='#':
            row = line.rstrip().split(' ')
            #labels = row[0]
            #words = row[1:]

            label_list = []
            if update:
                for label in row[0].split(','): 
                    if not label: continue
                    label_int = model.add_topic(label)
                    label_list.append(label_int)

            if len(row) <= 2: return None 

            word_list = []
            topic_count = {}
            for element in row[1:]:
                if element.count(':') == 1:
                    word, count = element.split(':')
                    count = max(1, min(10, int(math.floor(float(count))))) #int(count)
                else: 
                    word, count = element, 1
                if update:
                    word_int = model.add_word(word)
                else:
                    word_int = model.get_word(word)
                    if word_int < 0: 
                        continue

                for i in xrange(count): 
                    if model.word_seed_list[word_int] and random.randint(0, 1) == 0:
                        topic = model.word_seed_list[word_int][random.randint(0, len(model.word_seed_list[word_int])-1)]
                    elif label_list and random.randint(0, 1) == 0:
                        topic = label_list[random.randint(0, len(label_list)-1)] 
                    else:
                        topic = random.randint(0, model.topic_num-1)

                    word_list.append( [word_int, topic] )
                    topic_count[topic] = topic_count.get(topic, 0) + 1
                    if update:
                        model.topic_word_count[topic][word_int] = model.topic_word_count[topic].get(word_int, 0) + 1
                        model.topic_count[topic] += 1.
                        model.word_count[word_int] += 1. 
            doc = Document(word_list, label_list, topic_count)
            return doc
        else:
            return None

    def init_corpus_and_model(self, p_train, model): 
        for line in open(p_train):
            doc = Corpus.init_doc(line, model, update=True)
            if not doc: continue
            self.doc_list.append(doc)
            model.doc_num += 1
        print 'topics', model.topic_num
        print 'docs', len(self.doc_list)
        print 'words', len(model.word_seed_list)
            
