import pandas as pd
from scipy.stats import wilcoxon

df = pd.DataFrame()
data = pd.read_excel('homo_new_mean.xls').set_index('Unnamed: 0').sort_index()
our = data.loc[:,'DES-SV']
#data.to_excel('heter.xls')
print(data)
for comparison in data.columns[:-1]:
    print('DES-SV VS. ',comparison)
    print(wilcoxon(our,data.loc[:,comparison]))
    print('------------------------')
