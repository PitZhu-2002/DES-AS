import pandas as pd
from scipy.stats import friedmanchisquare

data = pd.read_excel('k_wtl.xlsx').set_index('Unnamed: 0')
stat_f2,p_f2 = friedmanchisquare(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],data.iloc[:,3],
                                 data.iloc[:,4], data.iloc[:,5],data.iloc[:,6])
print(stat_f2)
print(39*stat_f2/(39*6-stat_f2))
print(p_f2)
