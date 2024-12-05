# coding=utf-8

# Author: Yun-Hao Zhu <pit_yhzhu@163.com>
#
# License: BSD 3 clause


import copy
import numpy as np
from deslib.des.base import BaseDES
from scipy.stats._mstats_basic import mode
from sklearn.utils import shuffle


class DES_AS(BaseDES):
    def __init__(self, type,pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, mode='selection',pre_simulation = 100,
                 random_state=None, knn_classifier='knn', knne=False,
                 DSEL_perc=0.5, n_jobs=-1, voting='tough', needs_proba=False,convergence = 0.05):
        self.type = type
        self.voting = voting
        self.convergence = convergence
        self.rd = 0
        self.pre_simulation = pre_simulation
        super(DES_AS, self).__init__(pool_classifiers=pool_classifiers,
                                         k=k,
                                         DFP=DFP,
                                         with_IH=with_IH,
                                         safe_k=safe_k,
                                         IH_rate=IH_rate,
                                         mode=mode,
                                         random_state=random_state,
                                         knn_classifier=knn_classifier,
                                         knne=knne,
                                         DSEL_perc=DSEL_perc,
                                         n_jobs=n_jobs,
                                         needs_proba=needs_proba)

    def estimate_competence_from_proba(self, query, neighbors, distances=None, probabilities=None):
        competence = self.shapely_value(neighbors)
        return np.array(competence)

    def estimate_competence(self, query, neighbors, distances=None, predictions=None):
        competence = self.shapely_value(neighbors)
        return self.sv0_normalize(competence)

    def select(self, competences):
        selected_classifiers = competences >= 0
        selected_classifiers[~np.any(selected_classifiers, axis=1), :] = True
        return selected_classifiers
    def sv0_normalize(self, competence):
        normalization = []
        for sv in competence:
            sv = sv / np.sum(sv[sv >= 0])
            normalization.append(sv)
        return np.array(normalization)
    def pool_to_np(self,pool):
        num = 0
        compile = []
        for i in pool:
            num = num + 1
            compile.append(i)
        return np.array(compile), num
    def shapely_value(self, neighbors):
        SHAP = []   # 每个 neighbors 中基分类 Shapley Value 值
        pool,num = self.pool_to_np(self.pool_classifiers)   # pool_classifier 的 Numpy 形式, num 分类器数量
        for i in range(0, len(neighbors)):
            domain_x = self.DSEL_data_[neighbors[i]]        # Domain X of the existing test sample.
            domain_y = self.DSEL_target_[neighbors[i]] + 1  # Domain y of the existing test sample.
            idx = np.array([i for i in range(num)])         # classifier index
            shap_100_pm = []
            for t in range(100):
                shap_dic = {}
                pm = shuffle(idx,random_state = self.rd)
                self.rd = self.rd + 1
                for j,n in enumerate(pm):
                    prd = pool[n].predict(domain_x) + 1
                    if t == 0:  # 第一次在 dict 中放入
                        if j == 0:
                            shap_dic[pool[n]] = self.accuracy(prd,domain_y)
                        else:
                            acc_pxi = self.accuracy(self.ticket(self.Bagging(pool[pm[:j+1]],domain_x)),domain_y)
                            acc_p = self.accuracy(self.ticket(self.Bagging(pool[pm[:j]],domain_x)),domain_y)
                            shap_dic[pool[n]] = acc_pxi - acc_p
                    else:
                        if j == 0:
                            shap_dic[pool[n]] = self.accuracy(prd, domain_y) / (t + 1) + (t * shap_100_pm[-1][pool[n]]) / (t + 1)
                        else:
                            acc_pxi = self.accuracy(self.ticket(self.Bagging(pool[pm[:j+1]], domain_x)), domain_y)
                            acc_p = self.accuracy(self.ticket(self.Bagging(pool[pm[:j]],domain_x)),domain_y)
                            shap_dic[pool[n]] = (acc_pxi - acc_p) / (t + 1) + (t * shap_100_pm[-1][pool[n]]) / (t + 1)
                shap_100_pm.append(shap_dic)    # 长度为 100, 保存 0-99 次实验结
            t = self.pre_simulation
            while self.convergence_criterion(shap_100_pm):
                t = t + 1
                pm = shuffle(copy.deepcopy(idx),random_state = self.rd)
                self.rd = self.rd + 1
                shap_99 = {}
                for j,n in enumerate(pm):
                    prd = pool[n].predict(domain_x) + 1
                    if j == 0:
                        shap_99[pool[n]] = self.accuracy(prd,domain_y) / (t + 1) + t * shap_100_pm[-1][pool[n]] / (t + 1)
                    else:
                        acc_pxi = self.accuracy(self.ticket(self.Bagging(pool[pm[:j+1]], domain_x)), domain_y)
                        acc_p = self.accuracy(self.ticket(self.Bagging(pool[pm[:j]], domain_x)), domain_y)
                        shap_99[pool[n]] = (acc_pxi - acc_p) / (t + 1) + t * shap_100_pm[-1][pool[n]] / (t + 1)
                del shap_100_pm[0]
                shap_100_pm.append(shap_99)
            shap_value_tmc = []
            sv_100 = shap_100_pm[-1]
            for clf in pool:
                shap_value_tmc.append(sv_100[clf])
            SHAP.append(np.array(shap_value_tmc))
        return np.array(SHAP)
    def convergence_criterion(self,shap_100_pm):
        thita_t = shap_100_pm[-1]
        thita_t_100 = shap_100_pm[0]
        sum = 0
        for i in thita_t:
            if thita_t[i] == 0:
                continue
            else:
                sum = sum + abs(thita_t[i] - thita_t_100[i]) / abs(thita_t[i])
        return (sum / len(thita_t)) >= self.convergence
    def ticket(self, predict_sum):
        return mode(np.array(predict_sum),axis=0)[0][0]
    def accuracy(self, prd, Y_test):
        return np.sum(prd == Y_test) / len(Y_test)
    def Bagging(self, base_csf, X_test):
        bag_prediction = []
        for model in base_csf:
            bag_prediction.append(model.predict(X_test) + 1)
        return np.array(bag_prediction)
