from sklearn.naive_bayes import GaussianNB

import Consts
import pandas as pd

from modeling import Modeling


class Party:
    name = ''
    id = 0
    amount_of_voters = 0
    voters_percentage = 0
    strongest_features = list()     # type: [str]

    def __init__(self, name, id, amount_of_voters, voters_percentage):
        self.name = name
        self.id = id
        self.amount_of_voters = amount_of_voters
        self.voters_percentage = voters_percentage

    def report(self):
        msg = "\n"
        msg += f"Party:              {self.name}\n"
        msg += f"Amount of Voters:   {self.amount_of_voters}\n"
        msg += f"Voters percentage:  {self.voters_percentage}\n"
        msg += "Strongest Features:  {}\n".format(",".join(self.strongest_features))

        print(msg)


df = pd.read_csv(Consts.FileNames.RAW_FILE_PATH.value)

dict_parties = dict()
for party in Consts.MAP_VOTE_TO_NUMERIC.keys():
    name = party
    amount_of_voters = df[Consts.VOTE_STR][df[Consts.VOTE_STR] == party].count()
    voters_percentage = amount_of_voters / df.shape[0]
    id = Consts.MAP_VOTE_TO_NUMERIC[party]
    dict_parties[party] = Party(name, amount_of_voters, voters_percentage)

[party.report() for party in dict_parties.values()]


def is_a_coalition(list_parties: [int]) -> bool:
    if len(list_parties) == 0:
        return False

    sum = 0
    for party in list_parties:
        sum += dict_parties[party].voters_percentage

    return sum >= 0.51



class ImportantFeatures(Modeling):
    gnb = None

    def __init__(self):
        super().__init__()
        use_set = 1
        self.load_data(Consts.FileNames.FILTERED_AND_SCALED, use_set)
        self.gnb = GaussianNB()

    def fit_train(self):
        self.gnb.fit(self.dict_dfs_pd[Consts.FileSubNames.X_TRAIN], self.dict_dfs_pd[Consts.FileSubNames.Y_TRAIN])

    def fit_train_and_val(self):
        pass































