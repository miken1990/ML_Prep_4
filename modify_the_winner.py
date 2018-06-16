from sklearn.naive_bayes import GaussianNB

import Consts
from modeling import Logger

import numpy as np
import pandas as pd
from pandas import read_csv
from time import strftime

factor = -0.5


class ModifyTheWinner:
    dict_dfs_pd = {d: None for d in list(Consts.FileSubNames)}
    logger = None
    clf = None
    original_dist = None

    def __init__(self, file_str=None, print_modeling: bool = False):
        file_str = "ModifyTheWinner_" + strftime("%y_%m_%d_%H_%M_%S") + ".txt" if file_str is None else file_str
        self.logger = Logger(file_str, print_modeling)
        self.clf = GaussianNB()

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
            self.dict_dfs_pd[d] = self._load_data(file_location)

    def _load_data(self, filePath):
        return read_csv(filePath, header=0, keep_default_na=True)

    def fit_train(self):
        x = self.dict_dfs_pd[Consts.FileSubNames.X_TRAIN]
        x = x.drop(columns=[Consts.INDEX_COL])
        y = self.dict_dfs_pd[Consts.FileSubNames.Y_TRAIN]
        y = y[Consts.VOTE_STR]
        self.clf.fit(x, y)

    def predict_the_winner(self, test_data) -> None:

        y_pred = self.clf.predict(test_data)
        y = y_pred.astype(np.int32)
        counts = np.bincount(y)
        winner = Consts.MAP_NUMERIC_TO_VOTE[np.argmax(counts)]
        print(f"\nWinner {winner}\n")

    def new_test_modify_large_party_to_maybe(self) -> pd.DataFrame:
        print("modify_large_party_to")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()
        x[Consts.SelectedFeatures.Will_vote_only_large_party.value + '_int'] = \
            x[Consts.SelectedFeatures.Will_vote_only_large_party.value + '_int'] + factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Overall_happiness_score(self) -> pd.DataFrame:
        print("Overall_happiness_score")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        x[Consts.SelectedFeatures.Overall_happiness_score.value] = \
            x[Consts.SelectedFeatures.Overall_happiness_score.value] - factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Avg_Satisfaction_with_previous_vote(self) -> pd.DataFrame:
        print("Avg_Satisfaction_with_previous_vote")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        x[Consts.SelectedFeatures.Avg_Satisfaction_with_previous_vote.value] = \
            x[Consts.SelectedFeatures.Avg_Satisfaction_with_previous_vote.value] + factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Most_Important_Issue_Education(self) -> pd.DataFrame:
        print("Important_Issue_Education")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Education'] = \
            x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Education'] + factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Most_Important_Issue_Environment(self) -> pd.DataFrame:
        print("Important_Issue_Environment")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Environment'] = \
            x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Environment'] + factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Most_Important_Issue_Financial(self) -> pd.DataFrame:
        print("Important_Issue_Financialt")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Financial'] = \
            x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Financial'] + factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Most_Important_Issue_Foreign_Affairsl(self) -> pd.DataFrame:
        print("Important_Issue_Foreign_Affairs")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Foreign_Affairs'] = \
            x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Foreign_Affairs'] + factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Most_Important_Issue_Healthcare(self) -> pd.DataFrame:
        print("Important_Issue_Healthcares")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Healthcare'] = \
            x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Healthcare'] + factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Most_Important_Issue_Military(self) -> pd.DataFrame:
        print("Important_Issue_Military")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Military'] = \
            x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Military'] + factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Most_Important_Issue_Other(self) -> pd.DataFrame:
        print("Important_Issue_Other")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Other'] = \
            x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Other'] + factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Most_Important_Issue_Social(self) -> pd.DataFrame:
        print("Important_Issue_Social")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Social'] = \
            x[Consts.SelectedFeatures.Most_Important_Issue.value + '_Social'] + factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Number_of_valued_Kneset_members(self) -> pd.DataFrame:
        print("Number_of_valued_Kneset_members")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        x[Consts.SelectedFeatures.Number_of_valued_Kneset_members.value] = \
            x[Consts.SelectedFeatures.Number_of_valued_Kneset_members.value] + factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Weighted_education_rank(self) -> pd.DataFrame:
        print("Weighted_education_rank")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        x[Consts.SelectedFeatures.Weighted_education_rank.value] = \
            x[Consts.SelectedFeatures.Weighted_education_rank.value] + factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Yearly_IncomeK(self) -> pd.DataFrame:
        print("Yearly_IncomeK")
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        x[Consts.SelectedFeatures.Yearly_IncomeK.value] = \
            x[Consts.SelectedFeatures.Yearly_IncomeK.value] + factor
        x = x.drop(columns=[Consts.INDEX_COL])
        return x

    def new_test_Garden_sqr_meter_per_person_in_residancy_area(self) -> pd.DataFrame:
        print("Garden_sqr_meter_per_person_in_residancy_area")
        X = self.dict_dfs_pd[Consts.FileSubNames.X_VAL].copy()  # type: pd.DataFrame
        X[Consts.SelectedFeatures.Garden_sqr_meter_per_person_in_residancy_area.value] = \
            X[Consts.SelectedFeatures.Garden_sqr_meter_per_person_in_residancy_area.value] + factor
        X = X.drop(columns=[Consts.INDEX_COL])
        return X

    def predict_voters_distribution(self, test_data):
        y_pred = self.clf.predict(test_data)
        bins = np.bincount(y_pred)
        total_y = y_pred.shape[0]
        print("Party Distribution:")
        diff = (bins / total_y) - (self.original_dist)
        print(f"Turquoises differance is {diff[Consts.MAP_VOTE_TO_NUMERIC['Turquoises']]}")
        for i in range(1, 12):
            string_to_write = Consts.MAP_NUMERIC_TO_VOTE[i] + f': {bins[i] / total_y} \n\t different by: {diff[i]}'
            print(string_to_write)

        coalition = {'Browns', 'Purples', 'Whites', 'Greens', 'Pinks'}
        strength = 0
        for party in coalition:
            index = Consts.MAP_VOTE_TO_NUMERIC[party]
            strength += bins[index] / total_y

        print(f"New Coalition Strength: {strength}")
        # biffest_diff_party = Consts.MAP_NUMERIC_TO_VOTE[np.argmax(diff)]
        # good_dif = np.max(diff)
        # smallst_diff_party = Consts.MAP_NUMERIC_TO_VOTE[np.argmin(diff)]
        # bad_dif = np.min(diff)
        # print(f"Positive difference at {biffest_diff_party} by: {good_dif}")
        # print(f"Negative difference at {smallst_diff_party} by: {bad_dif}")
        # winner = Consts.MAP_NUMERIC_TO_VOTE[np.argmax(bins)]
        # print(f"Winner {winner}\n")

    def original_prediction(self):
        x = self.dict_dfs_pd[Consts.FileSubNames.X_VAL]
        x = x.drop(columns=[Consts.INDEX_COL])
        y_pred = self.clf.predict(x)
        bins = np.bincount(y_pred)
        self.original_dist = bins / y_pred.shape[0]


if __name__ == "__main__":
    mtw = ModifyTheWinner()
    mtw.load_data(Consts.FileNames.FILTERED_AND_SCALED, 1)
    mtw.fit_train()
    mtw.original_prediction()
    x = mtw.new_test_Avg_Satisfaction_with_previous_vote()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Yearly_IncomeK()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Weighted_education_rank()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Number_of_valued_Kneset_members()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Most_Important_Issue_Education()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Most_Important_Issue_Environment()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Most_Important_Issue_Financial()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Most_Important_Issue_Foreign_Affairsl()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Most_Important_Issue_Healthcare()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Most_Important_Issue_Military()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Most_Important_Issue_Other()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Most_Important_Issue_Social()
#    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Avg_Satisfaction_with_previous_vote()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_modify_large_party_to_maybe()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Garden_sqr_meter_per_person_in_residancy_area()
    mtw.predict_voters_distribution(x)

    x = mtw.new_test_Overall_happiness_score()
    mtw.predict_voters_distribution(x)
