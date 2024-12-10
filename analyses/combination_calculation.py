import itertools
import math
from copy import copy
from itertools import permutations
import numpy as np
import pandas as pd
import deslib.util.diversity
import deslib.des.des_knn

def ticket(value):
    situation = np.sum(value,axis = 0) / len(value)
    result = []
    for i in situation:
        if i == 0.5:
            result.append(3)
        elif i < 0.5:
            result.append(0)
        elif i > 0.5:
            result.append(1)
    result = np.array(result)
    return result

def accuracy(result,truth):
    result = np.array(result)
    acc = 0
    for i in range(len(result)):
        if result[i] == 3:
            acc = acc + 0.5
        elif result[i] == truth[i]:
            acc = acc + 1
    return acc / len(truth)

def permutation(value):
    li = []
    for i in range(len(value)):
        for j in range(i+1,len(value)):
            li.append((i,j))
    return li
def combination_(value):
    size = len(value)
    idx = [i for i in range(len(value))]
    dic = {}
    for j in range(1,size):
        dic[j] = set(itertools.combinations(idx,j))
    return dic
def DF(value,i,k):
    value = np.array(value)
    al1 = value[i]
    al2 = value[k]
    n_00,n_11,n_01,n_10 = 0,0,0,0
    for i in range(len(al1)):
        if al1[i] == 1:
            if al2[i] == 1:
                n_11 = n_11 + 1
            else:
                n_10 = n_10 + 1
        else:
            if al2[i] == 0:
                n_00 = n_00 + 1
            else:
                n_01 = n_01 + 1
    return n_00 / (n_01+n_10+n_11+n_00)
def LOO(value,location,truth):
    data = copy(value)
    del data[location]
    acc = accuracy(ticket(np.array(data)),truth=truth)
    return accuracy(ticket(np.array(value)),truth=truth) - acc

value = [
    [1,1,1,1],
    [1,0,0,0],
    [0,1,0,1],
    [1,0,1,0],
]

truth = np.array([1,1,1,1])



# value = [
#     [1,1,1,1,0],
#     [1,0,1,1,0],
#     [1,0,0,1,1],
#     [1,1,0,1,0],
#     [0,1,0,1,1]
# ]
# value = [
#     [1,1,1,1,0],
#     [1,0,1,1,0],
#     [1,0,0,1,1],
#     [1,1,0,1,0],
#     [0,1,0,1,1]
# ]
# truth = np.array([1,1,1])
# value = [
#     [1,1,0],
#     [0,1,1],
#     [0,0,0]
# ]

comb = combination_(value)
for i in comb.keys():
    for j in comb[i]:
        print(i,':',j)
    print('*************')
combination_value = [0 for i in range(len(value))]  # 记录各个分类器的值
sequence = [i for i in range(len(value))]
for i in sequence:      # i 表示 classifier 的下标
    count = 1
    combination_value[i] = np.sum(np.array(value[i]) == truth) / len(truth)
    #comb = copy(comb)
    for j in range(1, len(value)):
        # j 表示 前面有 n 个
        for combi in comb[j]:
            if i not in combi:
                count = count + 1
                idx = copy(list(combi))
                idx.append(i)
                idx = np.array(idx)
                combination_value[i] = combination_value[i] + \
                accuracy(ticket(np.array(value)[idx]),truth=truth) - \
                accuracy(ticket(np.array(value)[idx][:-1]), truth=truth)
    print(count)
    combination_value[i] = combination_value[i] / count
print(np.array(combination_value))



for i in range(len(value)):
    print('位置:',i,'LOO:',LOO(value,i,truth))
sequence = [i for i in range(len(value))]   # 编号顺序
dd = permutations(sequence)
dicn = {}
for i in range(len(value)):
    st = [0,1,2,3,4,5,6,7,8]
    dicn[st[i]] = 0
for per in dd:
    for i,data in enumerate(per):
        if i == 0:
            dicn[data] = dicn[data] + np.sum(np.array(value[data]) == truth) / len(truth)
        else:
            dicn[data] = dicn[data] + accuracy(ticket(np.array(value)[np.array(per)][:i+1]),truth=truth) - accuracy(ticket(np.array(value)[np.array(per)][:i]), truth=truth)
n = []
a = 0
for i in dicn.keys():
    v = dicn[i]/math.factorial(len(value))
    if v > 0:
        a = a + v
    n.append(dicn[i] / math.factorial(len(value)))
    print('分类器',i,'的SV_: ',dicn[i]/math.factorial(len(value)),'   分类器',i,'的Acc: ',accuracy(value[i],truth))
print(sum(n))
print(a)
