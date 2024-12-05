import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import BaggingClassifier
from deslib.util.aggregation import majority_voting


class BASE(ABC):
    @abstractmethod
    def __init__(self, pool, k=7, voting='hard', dsel_type='knn'):
        self.k = k
        self.pool = pool
        self.dsel_type = dsel_type
        self.voting = voting
        self.X_dsel = None
        self.y_dsel = None
        self.dsel_index = None

    @abstractmethod
    def competences(self, indices, distances):
        pass

    @abstractmethod
    def select(self, competences):
        pass

    def fit(self, X_dsel, y_dsel):
        self.X_dsel = X_dsel
        self.y_dsel = y_dsel
        return self

    def predict(self, X_test):
        distances, indices = self._find_dsel(X_test)
        competences = self.competences(indices, distances)
        selected_classifiers = self.select(competences)
        predictions = self._aggregate_predictions(X_test, selected_classifiers, competences)
        return predictions

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)

    def _find_dsel(self, X_test):
        nbrs = NearestNeighbors(n_neighbors=self.k)
        if self.dsel_type == 'knn':
            nbrs.fit(self.X_dsel)
        elif self.dsel_type == 'output_profiles':
            dsel_profiles = np.array([clf.predict(self.X_dsel) for clf in self.pool.estimators_]).T
            test_profiles = np.array([clf.predict(X_test) for clf in self.pool.estimators_]).T
            nbrs.fit(dsel_profiles)
        distances, indices = nbrs.kneighbors(test_profiles if self.dsel_type == 'output_profiles' else X_test)
        self.dsel_index = indices
        return distances, indices

    def _aggregate_predictions(self, X_test, selected_classifiers, competences):
        predictions = np.zeros((len(X_test), len(self.pool)), dtype=int)
        for i, classifiers in enumerate(selected_classifiers):
            active_classifiers = np.where(classifiers)[0]
            for clf_idx in active_classifiers:
                predictions[i, clf_idx] = self.pool.estimators_[clf_idx].predict([X_test[i]])[0] + 1

        if self.voting == 'hard':
            return mode(predictions, axis=1).mode[0]
        elif self.voting == 'soft':
            labels = np.unique(predictions)
            weighted_sums = np.array([np.sum(competences * (predictions == label), axis=1) for label in labels])
            return labels[np.argmax(weighted_sums, axis=0)]

    def _check_mask(self, mask):
        mask[np.all(mask == False, axis=1)] = True
        return mask