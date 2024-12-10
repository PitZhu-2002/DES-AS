# coding=utf-8

# Author: Yun-Hao Zhu <pit_yhzhu@163.com>
# License: BSD 3 clause

import copy
import numpy as np
from deslib.des.base import BaseDES
from scipy.stats import mode
from sklearn.utils import shuffle


class DES_AS(BaseDES):
    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, mode='selection', pre_simulation=100,
                 random_state=None, knn_classifier='knn', knne=False,
                 DSEL_perc=0.5, n_jobs=-1, voting='tough', needs_proba=False,
                 convergence=0.05, type=None):
        super(DES_AS, self).__init__(
            pool_classifiers=pool_classifiers, k=k, DFP=DFP, with_IH=with_IH,
            safe_k=safe_k, IH_rate=IH_rate, mode=mode, random_state=random_state,
            knn_classifier=knn_classifier, knne=knne, DSEL_perc=DSEL_perc,
            n_jobs=n_jobs, needs_proba=needs_proba
        )
        self.type = type
        self.voting = voting
        self.convergence = convergence
        self.pre_simulation = pre_simulation
        self.random_state_gen = np.random.RandomState(random_state)  # For reproducible shuffling

    def estimate_competence_from_proba(self, query, neighbors, distances=None, probabilities=None):
        return self.shapely_value(neighbors)

    def estimate_competence(self, query, neighbors, distances=None, predictions=None):
        competence = self.shapely_value(neighbors)
        return self.normalize_shapely_values(competence)

    def select(self, competences):
        # Ensure at least one classifier is selected per query
        selected_classifiers = competences >= 0
        selected_classifiers[~np.any(selected_classifiers, axis=1), :] = True
        return selected_classifiers

    def normalize_shapely_values(self, shapely_values):
        normalized_values = []
        for sv in shapely_values:
            sv_sum = np.sum(sv[sv >= 0])
            if sv_sum == 0:
                normalized_values.append(sv)
            else:
                normalized_values.append(sv / sv_sum)
        return np.array(normalized_values)

    def shapely_value(self, neighbors):
        shapley_values = []
        pool, num_classifiers = self.get_classifier_pool_as_array()

        for neighbor_indices in neighbors:
            domain_x = self.DSEL_data_[neighbor_indices]
            domain_y = self.DSEL_target_[neighbor_indices]

            shapley_dict = self.compute_shapley_values(pool, domain_x, domain_y, num_classifiers)
            shapley_values.append(np.array([shapley_dict[clf] for clf in pool]))

        return np.array(shapley_values)

    def compute_shapley_values(self, pool, domain_x, domain_y, num_classifiers):
        # Initialize variables for Shapley value computation
        shapley_dict = {clf: 0.0 for clf in pool}
        idx = np.arange(num_classifiers)

        for _ in range(self.pre_simulation):
            permuted_idx = shuffle(idx, random_state=self.random_state_gen.randint(0, 2 ** 32))
            for j, clf_idx in enumerate(permuted_idx):
                clf = pool[clf_idx]
                current_subset = pool[permuted_idx[:j + 1]]
                previous_subset = pool[permuted_idx[:j]]

                acc_with_clf = self.accuracy(self.bagging_predict(current_subset, domain_x), domain_y)
                acc_without_clf = self.accuracy(self.bagging_predict(previous_subset, domain_x), domain_y)

                shapley_dict[clf] += (acc_with_clf - acc_without_clf) / self.pre_simulation

        return shapley_dict

    def get_classifier_pool_as_array(self):
        return np.array(self.pool_classifiers), len(self.pool_classifiers)

    def bagging_predict(self, classifiers, X):
        predictions = [clf.predict(X) for clf in classifiers]
        return np.array(predictions)

    def accuracy(self, predictions, true_labels):
        return np.mean(np.array(predictions) == true_labels)

    def ticket(self, predictions):
        return mode(predictions, axis=0).mode[0]
