#coding: utf-8

class Model:
    
    def __init__(self):
        self.topic_num = 0 
        self.word_seed_list = []
        self.word_topic_list = []

        self.label2int = {}
        self.int2label = {}
        self.word2int = {}
        self.int2word = {}

    def init_model(self, num_topic):
        self.topic_num = num_topic

    def add_topic(self, label):
        if label not in self.label2int:
            label_int = len(self.label2int) 
            self.label2int[label] = label_int
            self.int2label[label_int] = label
            if label_int >= self.topic_num:
                self.topic_num += 1
        return self.label2int[label]

    def add_word(self, word):
        if word not in self.word2int:
            word_int = len(self.word2int)
            self.int2word[word_int] = word
            self.word2int[word] = word_int
            self.word_seed_list.append([])
            self.word_topic_list.append({})
        return self.word2int[word]

    def load_rules(self, p_rule):
        for line in open(p_rule):
            if not line.strip() or line.startswith('#'): continue
            row = line.rstrip().split(' ')

            label = row[0]
            label_int = self.add_topic(label)

            for word in row[1:]:
                word_int = self.add_word(word)
                if label_int not in self.word_seed_list[word_int]:
                    self.word_seed_list[word_int].append(label_int)


    def loglikelihood(self):
        pass

    def update_model(self, corpus):
        pass

    def load_model(self, p_model):
        pass

