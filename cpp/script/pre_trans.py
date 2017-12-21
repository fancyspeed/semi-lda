import sys
with open(sys.argv[2],'w') as fo: 
  for line in open(sys.argv[1]):
    arr = line.replace(':',' ').strip().split(' ')
    new_line = []
    for i in range(len(arr)):
        if i%2==0: v = arr[i] 
        if i%2==1: 
            for i in range(int(arr[i])):
                new_line.append(v)
    fo.write(' ' + ' '.join(new_line) + '\n')
                

