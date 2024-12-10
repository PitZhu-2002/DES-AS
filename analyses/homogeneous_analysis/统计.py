import os
import numpy as np
import pandas as pd

data_std = pd.DataFrame()
all = os.listdir('D:\\tool\Machine_Learning1\com\hdu\Dynamic Ensemble Selection\一审新增实验\同质\Result')
for names in all:
    data = pd.read_excel(
        'D:\\tool\Machine_Learning1\com\hdu\Dynamic Ensemble Selection\一审新增实验\同质\Result\\'
        + names
    )
    cln = data.columns[1:]
    data = data.drop(labels=['Unnamed: 0'],axis=1).iloc[:30,]
    for comp in cln:
        #data_std.loc[names[11:-4],comp] = data.loc[:,comp].std()
        data_std.loc[names[11:-4],comp] = data.loc[:,comp].mean()
        #data_std.loc[names[11:-4],comp] = data.loc[:,comp].std()
print(data_std.to_excel('homo_new_mean.xls'))