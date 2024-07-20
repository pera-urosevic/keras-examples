import joblib
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class Data:
    def __init__(self):
        self.data = pd.read_csv("./data/Student_Performance.csv").dropna()

        # init data
        self.x = pd.DataFrame()
        self.y = pd.DataFrame()
        self.preps = {}

        # save data histogram
        fig, axis = plt.subplots(5, 1, figsize=(16, 24))
        self.data.hist(ax=axis)
        plt.savefig("./data/data.png")

        # add cols
        self.__add_col_scaled_min_max("Hours Studied", self.x)
        self.__add_col_scaled_min_max("Sleep Hours", self.x)
        self.__add_col_scaled_min_max("Previous Scores", self.x)
        self.__add_col_scaled_min_max("Sample Question Papers Practiced", self.x)
        self.__add_col_scaled_min_max("Performance Index", self.y)

        # split
        train_x, test_x, train_y, test_y = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        # save
        train_x.to_csv("./data/train_x.csv", index=False)
        test_x.to_csv("./data/test_x.csv", index=False)
        train_y.to_csv("./data/train_y.csv", index=False)
        test_y.to_csv("./data/test_y.csv", index=False)
        joblib.dump(self.preps, f"./data/preps.save")

    def __add_col_scaled_min_max(self, col_name, df):
        col = self.data[[col_name]]
        prep = MinMaxScaler().fit(col)
        df.loc[:, [col_name]] = prep.transform(col)
        self.preps[col_name] = prep


def get_train_data():
    train_x = pd.read_csv("./data/train_x.csv")
    train_y = pd.read_csv("./data/train_y.csv")
    return train_x, train_y


def get_test_data():
    test_x = pd.read_csv("./data/test_x.csv")
    test_y = pd.read_csv("./data/test_y.csv")
    return test_x, test_y


def get_preprocessors():
    preps = joblib.load("./data/preps.save")
    return preps
