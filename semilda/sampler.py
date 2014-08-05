#coding: utf-8
import random
import math
from corpus import Corpus

class Sampler:

    def __init__(self, model):
        self.model = model

    def sample_corpus(self, corpus):
        for doc in corpus.doc_list:
            self.sample_doc(doc, update=True)

    def sample_test(self, p_test, p_output, burn_in, max_iter):
        fo = open(p_output, 'w')
        for line in open(p_test):
            doc = Corpus.init_doc(line, self.model, update=False) 
            if not doc: 
                fo.write('\n') 
            for i in range(burn_in):
                self.sample_doc(doc, update=False)
            for i in range(max_iter):
                self.sample_doc(doc, update=False)
                doc.accumulative()
            sort_list = sorted(doc.accu_topic_count.items(), key=lambda d:-d[1])
            result_list = ['%s:%s' % (self.model.int2label.get(k, 'anonymous_%s'%k), v) for k, v in sort_list]
            fo.write(' '.join(result_list) + '\n')
        fo.close()

    def sample_doc(self, doc, update=False):
        for i, pair in enumerate(doc.word_list):
            word, topic = pair

            doc.topic_count[topic] -= 1
            #old_topic = topic

            if update:
                self.model.topic_word_count[topic][word] -= 1
                self.model.topic_count[topic] -= 1

            topic = self.sample_word(word, doc)

            #doc.topic_count[old_topic] -= 1
            doc.topic_count[topic] = doc.topic_count.get(topic, 0) + 1 

            if update:
                self.model.topic_word_count[topic][word] = self.model.topic_word_count[topic].get(word, 0) + 1 
                self.model.topic_count[topic] += 1.
            doc.word_list[i][1] = topic 
            

    def sample_word(self, word, doc):
        if self.model.word_seed_list[word] and random.randint(0, 1) == 0:
            return self.model.word_seed_list[word][random.randint(0, len(self.model.word_seed_list[word]) - 1)]

        #elif doc.label_list and random.randint(0, 1) == 0:
        #    return doc.label_list[random.randint(0, len(doc.label_list)-1)]

        topic_probs = [] 
        for topic in range(self.model.topic_num): 
            topic_word = self.model.topic_word_count[topic].get(word, 0) + self.model.beta
            topic_total = self.model.topic_count[topic] + self.model.word_num * self.model.beta
            doc_topic = doc.topic_count.get(topic, 0) + self.model.alpha 

            #word_total = self.model.word_count[word] + self.model.doc_num * self.model.beta #!!!

            topic_p = topic_word * doc_topic / topic_total #/ word_total #!!!
            topic_probs.append(topic_p)

        for label in doc.label_list: #!!!
            topic_probs[label] *= 5 #!!!

        rand_p = random.random() * sum(topic_probs)
        sofar = 0
        for i, p in enumerate(topic_probs):
            sofar += p
            if sofar >= rand_p:
                return i
        return random.randint(0, self.model.topic_num - 1)
        

    def loglikelihood(self, corpus):
        # loglikelihood = 1/|d| * sum_d sum_w log sum_z p(w|z)p(z|d)
        # p(z|d) = (doc.topic_count[z] + alpha) / (doc.length + topic_num * alpha)
        # p(w|z) = (topic_word_count[z][w] + beta) / (topic_count[z] + word_num * beta)
        sum_d = 0
        for doc in corpus.doc_list:
            p_z_ds = [] 
            doc_total = len(doc.word_list) + self.model.topic_num * self.model.alpha
            for topic in range(self.model.topic_num):
                doc_topic = doc.topic_count.get(topic, 0) + self.model.alpha
                p_z_d = doc_topic / doc_total 
                p_z_ds.append( p_z_d )
            sum_w = 0
            for pair in doc.word_list:
                word = pair[0]
                topic_total = self.model.topic_count[topic] + self.model.word_num * self.model.beta
                sum_z = 0         
                for topic in range(self.model.topic_num):
                    topic_word = self.model.topic_word_count[topic].get(word, 0) + self.model.beta
                    p_w_z = topic_word / topic_total 
                    sum_z += p_w_z * p_z_ds[topic]
                sum_w += math.log(sum_z)
            sum_d += sum_w
        return sum_d 

