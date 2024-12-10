import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import os
def main(data,dataset,n_splits = 30,test_size = 0.1,dsel_size = 0.33,n_estimator = 8,names = None):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    container = [[] for i in range(len(names))] # 储存结果容器
    stf_split = StratifiedShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = 10)
    count = 0
    for train_index, test_index in stf_split.split(X, y):
        X_prime, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_prime, y_test = np.array(y)[train_index], np.array(y)[test_index]
        count = count + 1
        print(count)
        ss = StratifiedShuffleSplit(n_splits=1, test_size=dsel_size, random_state=10)
        bagging = BaggingClassifier(tree.DecisionTreeClassifier(criterion='entropy',
        min_samples_split = 10, min_samples_leaf=2, random_state=10),
        n_estimators = n_estimator, bootstrap=True, random_state=10)
        adaboost = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='entropy',
        min_samples_split = 10, min_samples_leaf=2, random_state=10),
        n_estimators = n_estimator, random_state=10)
        comparison_container = [bagging, adaboost]
        for tr, ds in ss.split(X_prime, y_prime):
            X_train, y_train = X_prime[tr], y_prime[tr]  # 训练样本
            for idx in range(len(comparison_container)):
                comparison_container[idx].fit(X_train,y_train)
                container[idx].append(comparison_container[idx].score(X_test, y_test))
    df = pd.DataFrame()
    for i in range(len(names)):
        df[names[i]] = container[i]
    df.to_excel('D:\\tool\Machine_Learning1\com\hdu\Dynamic Ensemble Selection\一审新增实验\同质\Static Combination\\comparison_' + dataset + '.xls')


folder_path = 'D:\\tool\Machine_Learning1\com\hdu\Dynamic Ensemble Selection\\3_Heterogeneous\DES-KNN_proba\Datasets used in the experiments'
files = os.listdir(folder_path)
i = 0
for name in files:
    i = i + 1
    print('Number:',i,'dataset')
    data = pd.read_excel('D:\\tool\Machine_Learning1\com\hdu\Dynamic Ensemble Selection\\3_Heterogeneous\DES-KNN_proba\Datasets used in the experiments\\'+ name + '\\' + name + '.xls')
    names = ['Bagging','Adaboost']
    main(data,n_splits = 30,n_estimator = 40,names=names,dataset = name)

