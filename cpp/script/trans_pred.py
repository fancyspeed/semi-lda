#coding: utf-8
topic_dict = {}
def load_topicmap(p_in):
    for line in open(p_in):
        id, name = line.strip().split(' ')
        topic_dict[int(id)] = name

def trans(p_in, p_out):
    with open(p_out, 'w') as fo:
        for line in open(p_in):
            topic_weight = {}
            row = line.strip().split(' ')
            for i, value in enumerate(row):
                if i not in topic_dict: continue
                topic_weight[topic_dict[i]] = float(value)
            sort_list = sorted(topic_weight.items(), key=lambda d:-d[1])
            new_line = ' '.join(['%s:%s' % (k, v) for k, v in sort_list])
            fo.write(new_line + '\n')

import sys
if len(sys.argv) < 4:
    sys.exit(1)
load_topicmap(sys.argv[1])
trans(sys.argv[2], sys.argv[3])
