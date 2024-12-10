import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,rankdata

df = pd.DataFrame()
data = pd.read_excel('wtl_homo.xlsx').set_index('Unnamed: 0').sort_index()
our = data.loc[:,'DES-SV']
#data.to_excel('heter.xls')
print(data)
for comparison in data.columns[:-1]:
    print('DES-SV VS. ',comparison)
    wilcoxon(our,data.loc[:,comparison])
    #dif = our - data.loc[:,comparison]
    #ranks = rankdata(np.abs(dif),method='average')
    #print(np.sum(ranks[dif > 0]))
    #print(np.sum(ranks[dif < 0]))
    print('------------------------')