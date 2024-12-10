import pandas as pd
from deslib.dcs import LCA, OLA, MLA, MCB
from deslib.des import KNORAE, KNORAU, DESKNN, DESP, DESKL, KNOP, METADES
from deslib.des.des_as_tmc import DES_SV_TMC
from deslib.static import StaticSelection, SingleBest
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import os


# 主函数，用于在给定数据集上训练和评估多个模型
def main(data, dataset_name, n_splits=30, test_size=0.1, dsel_size=0.33):
    """
    Parameters:
    - data: pandas DataFrame, 包含数据集的DataFrame。
    - dataset_name: str, 数据集的名称，用于结果保存。
    - n_splits: int, 交叉验证的拆分次数。
    - test_size: float, 测试集所占的比例。
    - dsel_size: float, 用于动态选择的训练集比例。
    """

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 初始化用于存储每个模型结果的列表
    results = [[] for _ in range(len(get_model_names()))]

    # 创建分层随机拆分器
    stf_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=10)

    # 遍历每个拆分
    for train_index, test_index in stf_split.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 进一步拆分训练集以获取动态选择集
        ss = StratifiedShuffleSplit(n_splits=1, test_size=dsel_size, random_state=10)
        train_index, dsel_index = next(ss.split(X_train, y_train))
        X_train_fit, X_dsel = X_train.iloc[train_index], X_train.iloc[dsel_index]
        y_train_fit, y_dsel = y_train.iloc[train_index], y_train.iloc[dsel_index]

        # 创建分类器池
        pool = create_classifier_pool()

        # 训练分类器池中的每个基分类器
        for classifier in pool:
            classifier.fit(X_train_fit, y_train_fit)

        # 初始化动态选择模型
        models = get_models(pool,dsel_size=len(y_dsel))

        # 训练和评估每个模型
        for model, result in zip(models, results):
            model.fit(X_dsel, y_dsel)
            result.append(model.score(X_test, y_test))

    # 将结果保存到Excel文件
    save_results(results, dataset_name)


# 创建分类器池的函数
def create_classifier_pool():
    """
    创建包含多种分类器的池。

    Returns:
    - pool: list, 训练好的分类器列表。
    """
    # ... (省略了具体实现，与原始代码相同)

    # 组合所有模型到一个池中

    # SVM models with different kernels and C values
    svm_models = [SVC(kernel=kernel, C=c, probability=True, random_state=10)
                  for kernel in ['rbf', 'poly', 'linear', 'sigmoid'] for c in [1, 10, 100]]

    # Decision Tree models with different criteria and parameters
    dt_models = [DecisionTreeClassifier(criterion=criterion, min_samples_split=min_split, min_samples_leaf=min_leaf,
                                        random_state=10)
                 for criterion, min_split, min_leaf in [('gini', 10, 2), ('gini', 10, 5), ('gini', 20, 2),
                                                        ('gini', 20, 5), ('entropy', 10, 2), ('entropy', 10, 5),
                                                        ('entropy', 20, 2), ('entropy', 20, 5)]]

    # Naive Bayes models
    nb_models = [GaussianNB(), ComplementNB()]

    # MLP models with different hidden layer sizes and activations
    mlp_models = [MLPClassifier(hidden_layer_sizes=sizes, max_iter=2000, activation=activation, random_state=10)
                  for sizes, activation in [((20,), 'identity'), ((20,), 'logistic'), ((20,), 'tanh'), ((20,), 'relu'),
                                            ((10, 10), 'identity'), ((10, 10), 'logistic'), ((10, 10), 'tanh'),
                                            ((10, 10), 'relu')]]

    # KNN models with different n_neighbors
    knn_models = [KNeighborsClassifier(n_neighbors=n, p=2) for n in [1, 3, 5, 7, 9]]

    # Logistic Regression models with different solvers
    lr_models = [LogisticRegression(solver=solver, random_state=10)
                 for solver in ['lbfgs', 'saga', 'liblinear', 'newton-cg', 'sag']]

    # classifier pool
    pool = svm_models + dt_models + nb_models + mlp_models + knn_models + lr_models
    return pool

# 获取模型列表的函数
def get_models(pool,dsel_size):
    """
    根据分类器池创建动态选择和静态选择模型列表。

    Parameters:
    - pool: list, 分类器池。

    Returns:
    - models: list, 动态选择和静态选择模型列表。
    """
    models = [
        DES_SV_TMC(pool_classifiers=pool, k=3, random_state=10, mode='hybrid'),
        DES_SV_TMC(pool_classifiers=pool, k=5, random_state=10, mode='hybrid'),
        DES_SV_TMC(pool_classifiers=pool, k=7, random_state=10, mode='hybrid'),
        DES_SV_TMC(pool_classifiers=pool, k=9, random_state=10, mode='hybrid'),
        DES_SV_TMC(pool_classifiers=pool, k=11, random_state=10, mode='hybrid'),
        DES_SV_TMC(pool_classifiers=pool, k=13, random_state=10, mode='hybrid'),
        DES_SV_TMC(pool_classifiers=pool, k=dsel_size, random_state=10, mode='hybrid')
    ]
    return models


# 获取模型名称列表的函数
def get_model_names():
    """
    返回模型名称列表，用于结果DataFrame的列名。

    Returns:
    - model_names: list, 模型名称列表。
    """
    return [
        "DESFH", "KNORAE", "KNORAU", "DESP", "DESKL", "KNOP", "METADES",
        "DESKNN_DF", "DESKNN_Q", "DESKNN_ratio", "OLA", "LCA", "MLA", "MCB",
        "SingleBest", "StaticSelection", "DES-AS_TMC"
    ]


# 保存结果的函数
def save_results(results, dataset_name):
    """
    将结果保存到Excel文件。

    Parameters:
    - results: list, 每个模型的结果列表。
    - dataset_name: str, 数据集的名称。
    """
    df = pd.DataFrame()
    model_names = get_model_names()

    # 将结果添加到DataFrame中
    for i, name in enumerate(model_names):
        df[name] = results[i]

    # 保存DataFrame到Excel文件
    try:
        df.to_excel(f'../result/heterogeneous/comparison_{dataset_name}.xlsx', index=False)
    except Exception as e:
        print(f"Error saving results for {dataset_name}: {e}")


# 主程序入口
if __name__ == '__main__':
    folder_path = '../datasets/transformed_datasets'
    files = os.listdir(folder_path)

    # 遍历文件夹中的每个文件
    for i, file_name in enumerate(files, start=1):
        print(f'Processing dataset {i}: {file_name}')
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_excel(file_path)
        main(data, file_name)