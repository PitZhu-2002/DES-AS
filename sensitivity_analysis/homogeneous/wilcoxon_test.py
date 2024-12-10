import pandas as pd
from deslib.dcs import MCB
from scipy.stats import wilcoxon

df = pd.DataFrame()
data = pd.read_excel('wtl_homo.xlsx').set_index('Unnamed: 0')
print(data)
our = data.loc[:,'DES-SV']
print(data)
for comparison in data.columns[:-1]:
    print('DES-SV VS. ',comparison)
    print(wilcoxon(our,data.loc[:,comparison]))
    print('------------------------')
