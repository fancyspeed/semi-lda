#!/usr/bin/env python

import sys
import getopt


def Usage():
    print 'usage:'
    print '-h,--help: print help message.'
    print '-o,--output: output path' 
    print '--k: num of topics'
    print '-a,--alpha: alpha'
    print '-b,--beta: beta'

def main(argv):
    try:
        opts, args = getopt.getopt(argv[1:], 
    except getopt.GetoptError, err:
        print repr(err)
        Usage()
        exit.exit(10
    for o, a in opts:
        if o in ('-h', '--help'):
            print Usage()
            exit(1)
        if o in ('-o', '--output'):
            print a
            exit(1)

if __name__ == '__main__':
    main(sys.argv)

    
    
