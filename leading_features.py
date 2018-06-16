import os
import sys
from time import strftime

import pandas as pd
from pandas import read_csv
from sklearn.naive_bayes import GaussianNB

import Consts


# **********************************************************************************************************************#


class Logger(object):
    list_out_streams = list()

    def __init__(self, file_str=None, print_modeling: bool=False):
        if not print_modeling:
            return

        self.list_out_streams.append(sys.stdout)
        self.list_out_streams.append(open(file_str, "a"))

    def write(self, message):
        [stream.write(message + '\n') for stream in self.list_out_streams]

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

# **********************************************************************************************************************#


class ImportantFeatures:
    dict_dfs_pd = {d: None for d in list(Consts.FileSubNames)}
    logger = None
    gnb = None
    saved_features = None

    def __init__(self, is_print: bool=True):

        file_str = "ImportantFeatures_" + strftime("%y_%m_%d_%H_%M_%S") + ".txt"
        self.logger = Logger(file_str, is_print)
        use_set = 1
        self.load_data(Consts.FileNames.FILTERED_AND_SCALED, use_set)

        self.gnb = GaussianNB()

    def title(self, msg, decorator='*', decorator_len=80):
        self.logger.write(decorator * decorator_len)
        self.logger.write('{}: {}'.format(strftime("%c"), msg))
        self.logger.write(decorator * decorator_len)

    def log(self, msg):
        self.logger.write('{}: {}'.format(strftime("%c"), msg))

    def load_data(self, base: Consts.FileNames, set_num: int) -> None:
        self.title(f"Loading the data from {base}")
        # load train features and labels
        for d in list(Consts.FileSubNames):
            file_location = base.value.format(set_num, d.value)
            self.log(f"Loading {file_location}")
            self.dict_dfs_pd[d] = self._load_data(file_location)

    def _load_data(self, filePath):
        return read_csv(filePath, header=0, keep_default_na=True)

    def fit_train(self):
        x = self.dict_dfs_pd[Consts.FileSubNames.X_TRAIN]
        x = x.drop(columns=[Consts.INDEX_COL])
        y = self.dict_dfs_pd[Consts.FileSubNames.Y_TRAIN]
        y = y[Consts.VOTE_STR]
        self.gnb.fit(x, y)

    def fit_train_and_val(self):

        x = pd.concat([self.dict_dfs_pd[Consts.FileSubNames.X_TRAIN], self.dict_dfs_pd[Consts.FileSubNames.X_VAL]])
        y = pd.concat([self.dict_dfs_pd[Consts.FileSubNames.Y_TRAIN], self.dict_dfs_pd[Consts.FileSubNames.Y_VAL]])
        y = y[Consts.VOTE_STR]
        x = x.drop(columns=[Consts.INDEX_COL])
        self.saved_features = x.keys()
        self.gnb.fit(x, y)

    def report(self):
        msg = "\n"
        self.title(self.gnb)
        self.title("Sigma")
        self.log(self.gnb.sigma_)
        self.title("Theta")
        self.log(self.gnb.theta_)
        self.title("Class Priors")
        self.log(self.gnb.class_prior_)
        self.title("Class Count")
        self.log(self.gnb.class_count_)
        self.title("Classes")
        self.log(self.gnb.classes_)

    def sigma_to_csv(self):
        file_name = Consts.EX4DirNames.NAIVE_BAYES.value + Consts.EX4FileNames.SIGMA.value
        with open(file_name, "w") as file:

            if self.saved_features is not None:
                file.write(",".join(self.saved_features))
                file.write("\n")

            for line in self.gnb.sigma_:
                file.write(",".join([str(x) for x in line]))
                file.write("\n")

    def theta_to_csv(self):
        file_name = Consts.EX4DirNames.NAIVE_BAYES.value + Consts.EX4FileNames.THETA.value
        with open(file_name, "w") as file:

            if self.saved_features is not None:
                file.write(",".join(self.saved_features))
                file.write("\n")

            for line in self.gnb.theta_:
                file.write(",".join([str(x) for x in line]))
                file.write("\n")


def ex4_create_files():
    for d in Consts.EX4DirNames:
        if not os.path.isdir(d.value):
            os.mkdir(d.value)


def search_features():
    ex4_create_files()
    impf = ImportantFeatures()
    impf.fit_train_and_val()
    impf.sigma_to_csv()
    impf.theta_to_csv()


if __name__ == '__main__':
    search_features()
