import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Consts
from time import time, strftime
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn import metrics, mixture
from modeling import Logger


class Clustering:
    dict_dfs_np = {d: None for d in list(Consts.FileSubNames)}
    dict_dfs_pd = {d: None for d in list(Consts.FileSubNames)}
    logger = None

    def __init__(self, file_str=None, print_modeling: bool=False):
        file_str = strftime("%y_%m_%d_%H_%M_%S") + ".txt" if file_str is None else file_str
        self.logger = Logger(file_str, print_modeling)
        self.k_results = {k: list() for k in range(2, Consts.maxClustersNum)}

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
        """Performs clustering on the training data and measures performance according to 6 different evaluation methods
        :param k: number of clusters
        :return: None
        """
        # clf = KMeans(n_clusters=k, n_init=50, max_iter=5000, tol=1e-3, random_state=Consts.listRandomStates[0])
        clf = mixture.GaussianMixture(n_components=k, max_iter=100, n_init=10, tol=1e-3,
                                      random_state=Consts.listRandomStates[0])
        clf.fit(self.dict_dfs_np[Consts.FileSubNames.X_TRAIN])
        y_pred = clf.predict(self.dict_dfs_np[Consts.FileSubNames.X_TRAIN])

        # save scoring of evaluation functions that require true labels
        self.k_results[k].append((Consts.ClusteringPerformanceMetrics.ADJUSTED_RAND_INDEX,
                             metrics.adjusted_rand_score(self.dict_dfs_np[Consts.FileSubNames.Y_TRAIN], y_pred)))

        self.k_results[k].append((Consts.ClusteringPerformanceMetrics.MUTUAL_INFORMATION,
                             metrics.adjusted_mutual_info_score(self.dict_dfs_np[Consts.FileSubNames.Y_TRAIN], y_pred)))

        self.k_results[k].append((Consts.ClusteringPerformanceMetrics.HOMOGENEITY,
                             metrics.homogeneity_score(self.dict_dfs_np[Consts.FileSubNames.Y_TRAIN], y_pred)))

        self.k_results[k].append((Consts.ClusteringPerformanceMetrics.COMPLETENESS,
                             metrics.completeness_score(self.dict_dfs_np[Consts.FileSubNames.Y_TRAIN], y_pred)))

        # self.k_results[k].append((Consts.ClusteringPerformanceMetrics.FOWLKES_MALLOWS_SCORE,
        #                      metrics.fowlkes_mallows_score(self.dict_dfs_np[Consts.FileSubNames.Y_TRAIN], y_pred)))

        # save scoring of evaluation functions that don't require true labels
        self.k_results[k].append((Consts.ClusteringPerformanceMetrics.SILHOUTTE_COEFF,
                                  metrics.silhouette_score(self.dict_dfs_np[Consts.FileSubNames.X_TRAIN],
                                                           y_pred)))
        self.k_results[k].append((Consts.ClusteringPerformanceMetrics.CALINSKI_HARABAZ_INDEX,
                                  metrics.calinski_harabaz_score(self.dict_dfs_np[Consts.FileSubNames.X_TRAIN],
                                                           y_pred)))


    def cluster_perform(self):
        """Performs clustering on the training data with increasing k (between 2 and 13) and stores performance of 6
        different evaluation methods for every k
        :return: None
        """
        for i in range(2, Consts.maxClustersNum):
            self.cluster_with_k(i)

    def plot_metric_score(self):
        """for each metric we plot its score as a function of k
        :return: None
        """
        x_axis = list(range(2, Consts.maxClustersNum))
        # rand_index_scores = self._get_score_by_metric(Consts.ClusteringPerformanceMetrics.ADJUSTED_RAND_INDEX)
        # plt.title(Consts.ClusteringPerformanceMetrics.ADJUSTED_RAND_INDEX.value)
        # plt.plot(x_axis, rand_index_scores, 'r')
        # plt.show()
        # mutual_info_scores = self._get_score_by_metric(Consts.ClusteringPerformanceMetrics.MUTUAL_INFORMATION)
        # plt.title(Consts.ClusteringPerformanceMetrics.MUTUAL_INFORMATION.value)
        # plt.plot(x_axis, mutual_info_scores, 'b')
        # plt.show()
        # homogeneity_scores = self._get_score_by_metric(Consts.ClusteringPerformanceMetrics.HOMOGENEITY)
        # plt.title(Consts.ClusteringPerformanceMetrics.HOMOGENEITY.value)
        # plt.plot(x_axis, homogeneity_scores, 'g')
        # plt.show()
        comepleteness_scores = self._get_score_by_metric(Consts.ClusteringPerformanceMetrics.COMPLETENESS)
        plt.title(Consts.ClusteringPerformanceMetrics.COMPLETENESS.value)
        plt.plot(x_axis, comepleteness_scores, 'y')
        plt.show()
        # fowlkes_mallows_scores = self._get_score_by_metric(Consts.ClusteringPerformanceMetrics.FOWLKES_MALLOWS_SCORE)
        # plt.title(Consts.ClusteringPerformanceMetrics.FOWLKES_MALLOWS_SCORE.value)
        # plt.plot(x_axis, fowlkes_mallows_scores, 'p')
        # plt.show()
        silhouette_scores = self._get_score_by_metric(Consts.ClusteringPerformanceMetrics.SILHOUTTE_COEFF)
        plt.title(Consts.ClusteringPerformanceMetrics.SILHOUTTE_COEFF.value)
        plt.plot(x_axis, silhouette_scores, 'p')
        plt.show()
        calinski_harabaz_scores = self._get_score_by_metric(Consts.ClusteringPerformanceMetrics.CALINSKI_HARABAZ_INDEX)
        plt.title(Consts.ClusteringPerformanceMetrics.CALINSKI_HARABAZ_INDEX.value)
        plt.plot(x_axis, calinski_harabaz_scores, 'p')
        plt.show()

    def _get_score_by_metric(self, score_type):
        scores = [score if metric == score_type else None for k in
                  range(2, Consts.maxClustersNum) for metric, score in self.k_results[k]]
        return [x for x in scores if x is not None]

    def find_coalition(self, k=2):
        # we use the hyper-parameters that were best and make clustering on the whole data
        # clf = KMeans(n_clusters=k, n_init=50, max_iter=5000, tol=1e-3, random_state=Consts.listRandomStates[0])
        clf = mixture.GaussianMixture(n_components=k, max_iter=200, n_init=10, tol=1e-4,
                                      random_state=Consts.listRandomStates[0])
        all_data = np.concatenate((self.dict_dfs_np[Consts.FileSubNames.X_TRAIN],
                                   self.dict_dfs_np[Consts.FileSubNames.X_VAL],
                                   self.dict_dfs_np[Consts.FileSubNames.X_TEST]), axis=0)

        all_labels = np.concatenate((self.dict_dfs_np[Consts.FileSubNames.Y_TRAIN],
                                   self.dict_dfs_np[Consts.FileSubNames.Y_VAL],
                                   self.dict_dfs_np[Consts.FileSubNames.Y_TEST]), axis=0)

        clf.fit(all_data)
        y_pred = clf.predict(all_data)
        # print(clf.means_)
        # print(np.linalg.norm(clf.means_[0]-clf.means_[1]))

        clusters = []
        for x in range(k):
            clusters.append({Consts.MAP_NUMERIC_TO_VOTE[x]: [0, 0] for x in range(1, Consts.numOfParties + 1)})

        # assign number of voters for each party that are in each cluster
        for i in range(10000):
            clusters[y_pred[i]][Consts.MAP_NUMERIC_TO_VOTE[all_labels[i]]][0] += 1

        vote, counts = np.unique(all_labels, return_counts=True)
        vote_count = dict(zip(vote, counts))
        for i in range(10000):
            clusters[y_pred[i]][Consts.MAP_NUMERIC_TO_VOTE[all_labels[i]]][1] = \
                int((clusters[y_pred[i]][Consts.MAP_NUMERIC_TO_VOTE[all_labels[i]]][0] * 100)
                    /(vote_count[all_labels[i]]))

        # print each cluster in the following format:
        # cluster<cluster_num> voters: <voters_num>-<voters percentage>
        # distribution: {<party name>: [<voters_num>, <voters percentage from party>]}
        # closest clusters: [<distance between mean points>, <number of cluster>]
        for cluster_num, cluster in enumerate(clusters):
            num_of_voters_in_cluster = sum(x[0] for x in cluster.values())
            clusters_means = clf.means_
            # clusters_means = clf.cluster_centers_
            # np.linalg.norm(clf.means_[0] - clf.means_[1]
            cluster_distances = [(np.linalg.norm(clusters_means[index]-clusters_means[cluster_num]), index) for
                                 index, cluster_centre in enumerate(clusters_means)
                                  if index != cluster_num]
            cluster_distances.sort(key=lambda tup: tup[0])
            print('cluster' + str(cluster_num) + ' voters: ' + str(num_of_voters_in_cluster) + '-' +
                  str((num_of_voters_in_cluster)*100/Consts.numOfVoters) + '%\ndistribution: ' + str(cluster) + '\n' +
                  'closest clusters: ' + str(cluster_distances))
            print('\n')

    def clusters_get_distance(self, clusters):
        """calculates closest clusters by order for each cluster in clusters
        :param clusters: list of clusters-dictionaries
        :return:
        """


if __name__ == '__main__':
    c = Clustering()
    dataset = 1
    c.load_data(Consts.FileNames.FILTERED_AND_SCALED, dataset)
    c.cluster_perform()
    c.plot_metric_score()
    c.find_coalition(2)


