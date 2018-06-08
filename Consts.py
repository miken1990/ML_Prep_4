from enum import Enum
import numpy as np
from sklearn.metrics import make_scorer, confusion_matrix, f1_score

inf = 10000
maxLeafNodes = 20
maxClustersNum = 11
INDEX_COL = 'index_col'
VOTE_STR = 'Vote'
VOTE_INT = 'Vote_int'

setSelectedFeatures = {
    'Number_of_valued_Kneset_members',
    'Yearly_IncomeK',
    'Overall_happiness_score',
    'Avg_Satisfaction_with_previous_vote',
    'Most_Important_Issue',
    'Will_vote_only_large_party',
    'Garden_sqr_meter_per_person_in_residancy_area',
    'Weighted_education_rank',
    INDEX_COL
}

_listSymbolicColumns = [
    'Most_Important_Issue',
    'Main_transportation',
    'Occupation'
]

listSymbolicColumns = [feature for feature in _listSymbolicColumns if feature in setSelectedFeatures]

_listNonNumeric = [
    'Most_Important_Issue',
    'Main_transportation',
    'Occupation',
    'Looking_at_poles_results',
    'Married',
    'Gender',
    'Voting_Time',
    'Financial_agenda_matters',
    'Will_vote_only_large_party',
    'Age_group'
]

listNonNumeric = [feature for feature in _listNonNumeric if feature in setSelectedFeatures]


_setNumericFeatures = {
    'Avg_monthly_expense_when_under_age_21',
    'AVG_lottary_expanses',
    'Avg_Satisfaction_with_previous_vote',
    'Garden_sqr_meter_per_person_in_residancy_area',
    'Financial_balance_score_(0-1)',
    '%Of_Household_Income',
    'Avg_government_satisfaction',
    'Avg_education_importance',
    'Avg_environmental_importance',
    'Avg_Residancy_Altitude',
    'Yearly_ExpensesK',
    '%Time_invested_in_work',
    'Yearly_IncomeK',
    'Avg_monthly_expense_on_pets_or_plants',
    'Avg_monthly_household_cost',
    'Phone_minutes_10_years',
    'Avg_size_per_room',
    'Weighted_education_rank',
    '%_satisfaction_financial_policy',
    'Avg_monthly_income_all_years',
    'Last_school_grades',
    'Number_of_differnt_parties_voted_for',
    'Political_interest_Total_Score',
    'Number_of_valued_Kneset_members',
    'Overall_happiness_score',
    'Num_of_kids_born_last_10_years',
    'Age_group_int',
    'Occupation_Satisfaction',
    'Will_vote_only_large_party_int'
}

setNumericFeatures = { feature for feature in _setNumericFeatures if feature in setSelectedFeatures}

_setGaussianFeatures ={
    'Avg_Satisfaction_with_previous_vote',
     'Garden_sqr_meter_per_person_in_residancy_area',
     'Yearly_IncomeK',
     'Avg_monthly_expense_on_pets_or_plants',
     'Avg_monthly_household_cost',
     'Phone_minutes_10_years',
     'Avg_size_per_room',
     'Weighted_education_rank',
     'Number_of_differnt_parties_voted_for',
     'Political_interest_Total_Score',
     'Overall_happiness_score',
     'Num_of_kids_born_last_10_years',
     'Number_of_valued_Kneset_members',
     'Avg_monthly_income_all_years',
     'AVG_lottary_expanses',
     'Avg_monthly_expense_when_under_age_21',
}
setGaussianFeatures = {feature for feature in _setGaussianFeatures if feature in setSelectedFeatures}

_setUniformFeatures = {
    'Occupation_Satisfaction',
    'Avg_government_satisfaction',
    'Avg_education_importance',
    'Avg_environmental_importance',
    'Avg_Residancy_Altitude',
    'Yearly_ExpensesK',
    '%Time_invested_in_work',
    "Last_school_grades",
    '%_satisfaction_financial_policy',
    '%Of_Household_Income'
}

setUniformFeatures = {feature for feature in _setUniformFeatures if feature in setSelectedFeatures}

_listFixNegateVals = [
    'Avg_monthly_expense_when_under_age_21',
    'AVG_lottary_expanses',
    'Avg_Satisfaction_with_previous_vote'
]

listFixNegateVals = [feature for feature in _listFixNegateVals if feature in setSelectedFeatures]

listAdditionalDataPreparation = ["validation", "test", ""]

listRandomStates = [376674226, 493026216, 404629562, 881225405]

MAP_VOTE_TO_NUMERIC = {
    'Greens': 10,
    'Pinks': 9,
    'Purples': 8,
    'Blues': 7,
    'Whites': 6,
    'Browns': 5,
    'Yellows': 4,
    'Reds': 3,
    'Turquoises': 2,
    'Greys': 1,
    'Oranges': 11
}

MAP_NUMERIC_TO_VOTE = {v: k for k, v in MAP_VOTE_TO_NUMERIC.items()}

class RandomGrid:
    # random forest grid
    n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=20)]
    max_features = ['auto', "log2"]
    max_depth = [int(x) for x in np.linspace(2, 12, 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [5, 11, 21, 31]
    bootstrap = [True]
    random_forest_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
                   'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}
    # decision tree grid
    criterion = ["gini", "entropy"]
    splitter = ["best", "random"]
    max_depth = [int(x) for x in np.linspace(2, 12, 11)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [5, 11, 21, 31]

    presort = [True, False]
    decision_tree_grid = {'criterion': criterion, 'splitter': splitter, 'max_depth': max_depth,
                          'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
                          'max_features': max_features, 'presort': presort}
    # SVM grid
    kernel = ['linear', 'rbf', 'poly']
    C = [0.1, 0.5, 1, 5]
    gamma = [0.1, 0.2, 0.5]
    gamma.append('auto')
    svc_grid = {'kernel': kernel, 'C': C, 'gamma': gamma}

    # KNN grid
    k_n_neighbors = list(range(1, 31))
    k_weight = ['uniform', 'distance']
    knn_grid = {'n_neighbors': k_n_neighbors, 'weights': k_weight}


class ClassifierType(Enum):
    DECISION_TREE = 'decision_tree'
    RANDOM_FOREST = 'random forest'

# **********************************************************************************************************************#

# Scoring functions:

def f1_not_binary_score(y_true, y_pred):
    return np.mean(f1_score(y_true, y_pred, average=None))

def winner_scoring_data(y_true, y_pred):
    confusion_mat = confusion_matrix(y_true, y_pred)
    winner_tup = np.unravel_index(np.argmax(confusion_mat, axis=None), confusion_mat.shape)
    if winner_tup[0] != winner_tup[1]:
        return 0, 0, 0, 0
    tp = confusion_mat[winner_tup]
    fp = np.sum(confusion_mat[:,winner_tup[0]]) - tp
    fn = np.sum(confusion_mat[winner_tup[0]]) - tp
    tn = np.sum(confusion_mat) - (tp + fn + fp)

    return tp, fp, fn, tn

def recall_winner_score(y_true, y_pred):
    tp, fp, fn, tn = winner_scoring_data(y_true, y_pred)
    if tp == 0:
        return 0
    return tp / (tp + fn)

def precision_winner_score(y_true, y_pred):
    tp, fp, fn, tn = winner_scoring_data(y_true, y_pred)
    if tp == 0:
        return 0
    return tp / (tp + fp)

def f1_winner_score(y_true, y_pred):
    tp, fp, fn, tn = winner_scoring_data(y_true, y_pred)
    if tp == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)

def distribution_score_aux(y_true, y_pred):
    confusion_mat = confusion_matrix(y_true, y_pred)
    sum_rows = np.sum(confusion_mat, axis=1)
    sum_cols = np.sum(confusion_mat, axis=0)
    return np.mean([min(x,y)/max(x,y) for x, y in zip(sum_rows, sum_cols)])


recall_winner = make_scorer(recall_winner_score)
precision_winner = make_scorer(precision_winner_score)
f1_winner = make_scorer(f1_winner_score)
distribution_score = make_scorer(distribution_score_aux)
# **********************************************************************************************************************#

class ScoreType(Enum):
    # Classification
    # F1 = 'f1'
    # F1_MACRO = 'f1_macro'
    # F1_MICRO = 'f1_micro'
    # F1_WEIGHTED = 'f1_weighted'
    ACCURACY = 'accuracy'

    # winner scoring:
    WINNER_RECALL = recall_winner
    WINNER_PRECISION = precision_winner
    WINNER_F1 = f1_winner
    # distribution:
    DISTRIBUTION = distribution_score
    # Clustering

    # Regression
    # EXPLAINED_VARIANCE = 'explained_variance'
    # R2 = 'r2'


class ClassifierTypes(Enum):
    TREE = "tree"
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    KNN = 'knn'
    NAIVE_BAYES = 'naive bayes'


# **********************************************************************************************************************#

class DataTypes(Enum):
    TEST = 'test'
    VAL = 'val'
    TRAIN = 'train'


class FileSubNames(Enum):
    X_TRAIN = 'X_train'
    X_VAL = 'X_val'
    X_TEST = 'X_test'
    Y_TRAIN = 'Y_train'
    Y_VAL = 'Y_val'
    Y_TEST = 'Y_test'

class DirNames(Enum):
    DATA_SETS = 'datasets'
    DATA_SETS_I = 'datasets/{}'
    RAW_AND_SPLITED = "datasets/{}/raw_spited"
    RAW_AND_FILTERED = "datasets/{}/raw_and_filtered"
    FILTERED_AND_NUMERIC_NAN = "datasets/{}/filtered_and_numeric_nan"
    FILTERED_AND_NUMERIC_NONAN = "datasets/{}/filtered_and_numeric_nonan"
    FILTERED_AND_SCALED = "datasets/{}/filtered_and_scaled"
    SUMMARY = "datasets/{}/summary"

class EX3DirNames(Enum):
    BASE = 'EX3_data'
    SINGLE_ESTIMATOR = 'EX3_data/single_estimator'
    MULTI_ESTIMATORS = 'EX3_data/multi_estimators'
    SUMMARY = 'EX3_data/summary'

class EX3FilNames(Enum):
    WINNER = '/the_winner.csv'
    PREDICTED_DISTRIBUTION = '/predicted_distribution.txt'
    MOST_LIKELY_PARTY = '/most_likely_to_vote.txt'        # .format the name of the party
    CONFUSION_MATRIX = '/confusion_matrix.csv'

per_file = "/{}.csv"

class FileNames(Enum):
    FROM_INTERNET = ""
    RAW_FILE_PATH = 'ElectionsData.csv'
    RAW_AND_SPLITED = DirNames.RAW_AND_SPLITED.value + per_file
    RAW_AND_FILTERED  = DirNames.RAW_AND_FILTERED.value + per_file
    FILTERED_AND_NUMERIC_NAN = DirNames.FILTERED_AND_NUMERIC_NAN.value + per_file
    FILTERED_AND_NUMERIC_NONAN = DirNames.FILTERED_AND_NUMERIC_NONAN.value + per_file
    FILTERED_AND_SCALED = DirNames.FILTERED_AND_SCALED.value + per_file
    SUMMARY = DirNames.SUMMARY.value + per_file

class ClusteringPerformanceMetrics(Enum):
    ADJUSTED_RAND_INDEX = "Adjusted Rand Index"
    MUTUAL_INFORMATION = "Mutual Information"
