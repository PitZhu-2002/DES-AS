import numpy as np
import pandas as pd

data = pd.read_excel('homo_new_mean.xls').set_index('Unnamed: 0')
size = len(data.columns)
idx = data.index
df = pd.DataFrame()
df2 = pd.DataFrame()
for a in range(size):
    way = data.columns[a]
    win = 0
    tie = 0
    los = 0
    count = 0
    wtl_lst = []
    for name in idx:
        count = count + 1
        if data.loc[name,'DES-SV'] == data.loc[name,way]:
            wtl_lst.append('T')
            tie = tie + 1
        elif data.loc[name,'DES-SV'] > data.loc[name,way]:
            wtl_lst.append('W')
            win = win + 1
        else:
            wtl_lst.append('L')
            los = los + 1
    wtl_lst = np.array(wtl_lst)
    df2.loc[:,data.columns[a]] = wtl_lst
    result = [win,tie,los]
    print(result)
    three = ['win','tie','loss']
    for n in range(len(three)):
        df.loc[way,three[n]] = result[n]
print(df.to_excel('homo_new_wtl.xls'))
print(df2)
#df2.to_excel('wtl_form_form.xls')

