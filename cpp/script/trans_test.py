#coding: utf-8

def trans(p_in, p_out):
    fo = open(p_out, 'w')
    for line in open(p_in):
        row = line.rstrip().split(' ')
        word_count = {}
        for word in row[1:]:
            if word:
                word_count[word] = word_count.get(word, 0) + 1
        new_line = ' '.join(['%s %s' % (k, v) for k, v in word_count.items()])
        fo.write(new_line + '\n')
    fo.close()

import sys
if len(sys.argv) < 3:
    sys.exit(1)
trans(sys.argv[1], sys.argv[2])
