#coding: utf-8

class Model:
    
    def __init__(self):
        self.word_seed_list = []
        self.topic_word_count = [] 
        self.topic_count = [] 
        self.word_count = []

        self.label2int = {}
        self.int2label = {}
        self.word2int = {}
        self.int2word = {}

        self.accu_topic_word_count = []
        self.accu_topic_count = []

    def init_model(self, args):
        self.alpha = args.alpha
        self.beta = args.beta
        self.topic_num = args.num_topic
        self.word_num = 0
        self.doc_num = 0

        for i in range(args.num_topic):
            self.topic_word_count.append({})
            self.topic_count.append(0)
            self.accu_topic_word_count.append({})
            self.accu_topic_count.append(0)

    def add_topic(self, label):
        if label not in self.label2int:
            label_int = len(self.label2int) 
            self.label2int[label] = label_int
            self.int2label[label_int] = label
            if label_int >= self.topic_num:
                self.topic_num += 1
                self.topic_word_count.append({})
                self.topic_count.append(0)
                self.accu_topic_word_count.append({})
                self.accu_topic_count.append(0)
        return self.label2int[label]

    def add_word(self, word):
        if word not in self.word2int:
            word_int = len(self.word2int)
            self.int2word[word_int] = word
            self.word2int[word] = word_int
            self.word_seed_list.append([])
            self.word_count.append(0)
            self.word_num += 1
        return self.word2int[word]

    def get_word(self, word):
        return self.word2int.get(word, -1)

    def load_rules(self, p_rule):
        for line in open(p_rule):
            if not line.strip() or line.startswith('#'): continue
            row = line.rstrip().split(' ')
            if len(row) != 2: continue 

            label = row[0]
            if not label: continue
            label_int = self.add_topic(label)

            for word in row[1].split(','):
                if not word: continue
                word_int = self.add_word(word)
                if label_int not in self.word_seed_list[word_int]:
                    self.word_seed_list[word_int].append(label_int)


    def accumulative(self):
        for topic, word_count in enumerate(self.topic_word_count):
            for word in word_count:
                count = word_count[word]
                self.accu_topic_word_count[topic][word] = self.accu_topic_word_count[topic].get(word, 0) + count
            self.accu_topic_count[topic] += self.topic_count[topic]

    def load_model(self, p_model):
        fin = open(p_model)
        
        self.alpha, self.beta, self.topic_num, self.word_num, self.doc_num = eval(fin.readline().rstrip())
        self.word_seed_list = eval(fin.readline().rstrip())
        self.topic_word_count = eval(fin.readline().rstrip())
        self.topic_count = eval(fin.readline().rstrip())
        self.word_count = eval(fin.readline().rstrip())
        self.label2int = eval(fin.readline().rstrip())
        self.int2label = eval(fin.readline().rstrip())
        self.word2int = eval(fin.readline().rstrip())
        self.int2word = eval(fin.readline().rstrip())

        fin.close()

    def save_model(self, p_model):
        fo = open(p_model, 'w')

        fo.write(repr([self.alpha, self.beta, self.topic_num, self.word_num, self.doc_num]) + '\n')
        fo.write(repr(self.word_seed_list) + '\n')
        fo.write(repr(self.accu_topic_word_count) + '\n')
        fo.write(repr(self.accu_topic_count) + '\n')
        fo.write(repr(self.word_count) + '\n')
        fo.write(repr(self.label2int) + '\n')
        fo.write(repr(self.int2label) + '\n')
        fo.write(repr(self.word2int) + '\n')
        fo.write(repr(self.int2word) + '\n')

        fo.close()

    def dump_topic_words(self, p_dump):
        fo = open(p_dump, 'w')
        for topic in range(self.topic_num):
            word_count = self.accu_topic_word_count[topic]
            sort_list = sorted(word_count.items(), key=lambda d:-d[1])
            result_list = ['%s:%s' % (self.int2word[k], v) for k, v in sort_list]
            topic_name = self.int2label.get(topic, 'anonymous_%s'%topic)
            fo.write('%s %s\n' % (topic_name, ' '.join(result_list)))
        fo.close()

