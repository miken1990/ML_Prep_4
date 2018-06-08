import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

import Consts
from time import time, strftime
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn import metrics
from modeling import Logger


class Clustering:
    dict_dfs_np = {d: None for d in list(Consts.FileSubNames)}
    dict_dfs_pd = {d: None for d in list(Consts.FileSubNames)}
    logger = None

    def __init__(self, file_str=None, print_modeling: bool=False):
        file_str = strftime("%y_%m_%d_%H_%M_%S") + ".txt" if file_str is None else file_str
        self.logger = Logger(file_str, print_modeling)
        self.k_results = {k: list() for k in range(Consts.maxClustersNum)}

    def title(self, msg, decorator='*', decorator_len=80):
        self.logger.write(decorator * decorator_len)
        self.logger.write('{}: {}'.format(strftime("%c"), msg))
        self.logger.write(decorator * decorator_len)

    def log(self, msg):
        self.logger.write('{}: {}'.format(strftime("%c"), msg))

    def load_data(self, base: Consts.FileNames, set: int) -> None:
        """
        this method will load ready to use data for the training, validating, and testing sets.
        this implements stages 1, 3 and part of 6 in the assignment.
        :return:
        """
        self.title(f"Loading the data from {base}")
        # load train features and labels
        for d in list(Consts.FileSubNames):
            file_location = base.value.format(set, d.value)
            self.log(f"Loading {file_location}")
            if d in {Consts.FileSubNames.Y_TEST, Consts.FileSubNames.Y_VAL, Consts.FileSubNames.Y_TRAIN}:
                self.dict_dfs_np[d] = self._load_data(file_location)[Consts.VOTE_STR].as_matrix().ravel()
            else:
                self.dict_dfs_np[d] = self._load_data(file_location).as_matrix()
            self.dict_dfs_pd[d] = self._load_data(file_location)

    def _load_data(self, filePath):
        return read_csv(filePath, header=0, keep_default_na=True)

    def cluster_with_k(self, k):
        """Performs clustering on the training data
        :param k: number of clusters
        :return:
        """
        clf = KMeans(n_clusters=k)
        clf.fit(self.dict_dfs_np[Consts.FileSubNames.X_TRAIN.value])
        y_pred = clf.predict(self.dict_dfs_np[Consts.FileSubNames.X_TRAIN.value])
        self.k_results[k].append((Consts.ClusteringPerformanceMetrics.ADJUSTED_RAND_INDEX.value,
                             metrics.adjusted_rand_score(self.dict_dfs_np[Consts.FileSubNames.Y_TRAIN], y_pred)))


    def load_test_set(self):
        """
        this stage is done earlier when all the data was loaded.
        :return:
        """
        pass


    def apply_test_to_estimator(self, estimator):
        """
        apply the test set to the chosen estimator.
        TODO: do we want to pass the estimator or are we going to save it in self.
        TODO: do we want to return the predicted labels or save them? (I think returning is better)
        :return:
        """
        X_test = self.dict_dfs_pd[Consts.FileSubNames.X_TEST]
        return estimator.predict(X_test)

    def apply_test_to_cluster(self, cluster_estimator):
        """
        apply the test set to the chosen cluster.
        TODO: do we want to pass the cluster or are we going to save it in self.
        TODO: do we want to return the predicted labels or save them? (I think returning is better)
        :return:
        """
        X_test = self.dict_dfs_pd[Consts.FileSubNames.X_TEST]
        return cluster_estimator.predict(X_test)

    def check_performance(self, labels_pred):
        """
        how will we check performance?
            this is something we get to chose. there is no right answer.
            we need to define what is important to us between the cluster and estimator, and decide on a scoring function

        TODO: Decide on a scoring function (per each compared to label or compared between 2 predictions?)
        TODO: Decide how params will be passed, inside the class or as parameters.
        TODO: can we plot some kind of graph?
        :return:
        """
        labels_true = self.dict_dfs_np[Consts.FileSubNames.Y_TEST]
        return metrics.adjusted_rand_score(labels_true, labels_pred)

    def identify_leading_features_for_party(self, party):
        """
        identify the leading features for the given party.
        TODO: use the data in self or are we going to pass it as a parameter \ argument?
        :param party:
        :return:
        """
        # https://stackoverflow.com/questions/27491197/scikit-learn-finding-the-features-that-contribute-to-each-kmeans-cluster
        pass

    def identify_leading_features_for_all_parties(self):
        """
        identify the leading features for all the parties.
        loop over all parties and call identify_leading_features_for_party
        """
        pass

    def calculate_a_steady_coalition(self):
        """
        this does not need to be automated. better if it is.
        :return:
        """
        pass




gnb = sklearn.naive_bayes.GaussianNB()
