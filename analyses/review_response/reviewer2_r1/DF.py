import numpy as np


def permutation(value):
    li = []
    for i in range(len(value)):
        for j in range(i+1,len(value)):
            li.append((i,j))
    return li
def DF(value,i,k,label):
    value = np.array(value)
    al1 = value[i]
    al2 = value[k]
    n_00,n_11,n_01,n_10 = 0,0,0,0
    for i in range(len(al1)):
        if al1[i] != label[i] and al2[i] != label[i]:
            n_00 = n_00 + 1
    return n_00 / len(label)


truth = np.array([1,0,1,0,0])

value = [
    [1,1,1,0,0],
    [1,1,1,1,0],
    [1,0,0,1,0],
    [1,0,1,0,0],
    [0,1,1,1,1],
    [0,1,0,1,1]
]

name = [i for i in range(len(value))]
permu = permutation(name)
dic = {}

for each in permu:
    dic[each] = DF(value,each[0],each[1],truth)
print(dic)
result = [0 for i in range(len(value))]
for i in range(len(value)):
    for key in dic.keys():
        if i in key:
            result[i] = result[i] + dic[key]
print(np.array(result)/(len(value)-1))

