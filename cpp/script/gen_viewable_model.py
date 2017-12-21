#!/usr/bin/python
# coding: utf-8
# @author: zuotaoliu@126.com
# @created: 2013-12-28
import os
import sys

def gen(p_in, p_out):
    num_topics = 0
    map = []
    sum = []
    word_sum = {}

    for line in open(p_in):
      sep = line.split("\t")
      word = sep[0]
      sep = sep[1].split()

      if num_topics == 0:
        num_topics = len(sep)
        for i in range(num_topics):
          map.append({})
          sum.append(0.0)

      for i in range(len(sep)):
        if float(sep[i]) > 1:
          map[i][word] = float(sep[i])
          if word_sum.has_key(word):
            word_sum[word] += float(sep[i])
          else:
            word_sum[word] = float(sep[i])
          sum[i] += float(sep[i])
    
    for i in range(len(map)):
      for key in map[i].keys():
        map[i][key] = map[i][key] #/ word_sum[word]
    
    f_out = open(p_out, 'w')
    for i in range(len(map)):
      x = sorted(map[i].items(), key=lambda(k, v):(v, k), reverse = True)
      f_out.write('\n')
      f_out.write('TOPIC: %s %s\n' % (i, sum[i]))
      f_out.write('\n')
      for key in x:
        f_out.write('%s %s\n' % (key[0], key[1]))
    f_out.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print '<usage> input output'
        exit(-1)
    gen(sys.argv[1], sys.argv[2])

