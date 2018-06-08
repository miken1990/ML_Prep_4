import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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




