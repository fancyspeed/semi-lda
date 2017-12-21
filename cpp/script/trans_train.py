#!/usr/bin/python
# coding: utf-8
# @author: zuotaoliu@126.com
# @created: 2014-08-29
import os
import sys
import re

def do_word_index(p_in, p_out):
    word_count = {}
    topic_map = {}
    cur_idx = 0
    fo = open(p_out, 'w')
    for line in open(p_in):
        row = line.rstrip().split(' ')
        if len(row) <= 1: continue
        wc = {}
        for word in row[1:]:
            wc[word] = wc.get(word, 0) + 1
            word_count[word] = word_count.get(word, 0) + 1
        if row[0]:
            label = row[0]
            if label not in topic_map:
                topic_map[label] = cur_idx
                cur_idx += 1
            fo.write('[%s] %s\n' % (topic_map[label], ' '.join(['%s %s' % (k, v) for k, v in wc.items()]))) 
        else:
            fo.write('%s\n' % (' '.join(['%s %s' % (k, v) for k, v in wc.items()]))) 
    fo.close()
    return word_count, topic_map
        
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print '<usage> inputfile outputfile topicmap wordindex'
        exit(-1)
    word_count, topic_map = do_word_index(sys.argv[1], sys.argv[2])

    with open(sys.argv[3], 'w') as fo:
        for label in topic_map:
            fo.write('%s %s\n' % (topic_map[label], label))

    sort_list = sorted(word_count.items(), key=lambda d:d[1], reverse=True)
    with open(sys.argv[4], 'w') as fo:
        for id, pair in enumerate(sort_list):
            word, num = pair
            if num >= 20:
                fo.write('%s %s\n' % (id, word))


