import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from base import BASE


class DESP(BASE):

    def __init__(self, pool_classifiers, k=7, votings='hard'):
        super(DESP, self).__init__(pool_classifiers=pool_classifiers, k=k, votings=votings)

    def _confidence_level(self, competence_region, X_dsel, y_dsel, classifiers):
        '''
        :param index: the index of k nearest neighbors in X_dsel.
        :param classifiers: pool of classifiers.
        :return: row -> classifier columns -> region of competence.
        '''
        confidences = np.zeros((len(classifiers), self.k), dtype=float)
        X_roc, y_roc = X_dsel[competence_region], y_dsel[competence_region]
        for i, cls in enumerate(classifiers):
            proba = cls.predict_proba(X_roc)
            confidences[i] = proba[np.arange(self.k), y_roc - 1]
        return confidences

    def competences(self, competence_region):
        print(competence_region)
        # Container storing the computation results .
        container = np.zeros((len(competence_region), len(self.pool_classifiers)), dtype=float)
        for i, index in enumerate(competence_region):
            X_roc, y_roc = self.X_dsel[index], self.y_dsel[index]
            container[i] = self._output_acc(X_roc, y_roc, self.pool_classifiers)
        return container


    def select(self, competences):
        number = len(set(self.y_dsel))
        selected_classifiers = (competences > 1 / number)
        return selected_classifiers


if __name__ == '__main__':
    Datasets = 'bal'
    path = 'D:\\Software\\Programming\\Python Project\\pythonProject\\Machine learning datasets\\Tabular datasets\\Standardized datasets\\Clean\\'
    fpath = path + Datasets + '\\' + Datasets + '.xlsx'
    data = pd.read_excel(fpath)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    container = []
    container2 = []
    for i in range(30):
        X_prime, X_test, y_prime, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_dsel, y_train, y_dsel = train_test_split(X_prime, y_prime, test_size=0.25)
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_dsel, y_dsel = np.array(X_dsel), np.array(y_dsel)
        c = BaggingClassifier(
            estimator=tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=10, min_samples_leaf=2,
                                                  random_state=10), n_estimators=10, bootstrap=True).fit(X_train,
                                                                                                         y_train)
        spl = DESP(k=7, pool_classifiers=c).fit(X_dsel, y_dsel)
        container.append(spl.score(X_test, y_test))
    print(np.mean(container))
    print(np.mean(container2))


