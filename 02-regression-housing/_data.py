import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler


def prepare_data():
    data = pd.read_csv("./data/california_housing.csv")

    x = data.loc[:, ["longitude", "latitude", "median_income", "population"]]
    y = data.loc[:, ["median_house_value"]]

    longitude = x.loc[:, ["longitude"]]
    prep_longitude = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform", subsample=None).fit(longitude)
    x["longitude"] = prep_longitude.transform(longitude)
    joblib.dump(prep_longitude, "./data/prep_longitude.save")

    latitude = x.loc[:, ["latitude"]]
    prep_latitude = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform", subsample=None).fit(latitude)

    x["latitude"] = prep_latitude.transform(latitude)
    joblib.dump(prep_latitude, "./data/prep_latitude.save")

    median_income = x.loc[:, ["median_income"]]
    prep_median_income = MinMaxScaler().fit(median_income)
    x["median_income"] = prep_median_income.transform(median_income)
    joblib.dump(prep_median_income, "./data/prep_median_income.save")

    population = x.loc[:, ["population"]]
    prep_population = MinMaxScaler().fit(population)
    x["population"] = prep_population.transform(population)
    joblib.dump(prep_population, "./data/prep_population.save")

    median_house_value = y.loc[:, ["median_house_value"]]
    prep_median_house_value = MinMaxScaler().fit(median_house_value)
    y["median_house_value"] = prep_median_house_value.transform(median_house_value)
    joblib.dump(prep_median_house_value, "./data/prep_median_house_value.save")

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    train_x.to_csv("./data/train_x.csv", index=False)
    test_x.to_csv("./data/test_x.csv", index=False)
    train_y.to_csv("./data/train_y.csv", index=False)
    test_y.to_csv("./data/test_y.csv", index=False)


def get_train_data():
    train_x = pd.read_csv("./data/train_x.csv")
    train_y = pd.read_csv("./data/train_y.csv")
    return train_x, train_y


def get_test_data():
    test_x = pd.read_csv("./data/test_x.csv")
    test_y = pd.read_csv("./data/test_y.csv")
    return test_x, test_y


def get_preprocessors():
    prep_longitude = joblib.load("./data/prep_longitude.save")
    prep_latitude = joblib.load("./data/prep_latitude.save")
    prep_median_income = joblib.load("./data/prep_median_income.save")
    prep_population = joblib.load("./data/prep_population.save")
    prep_median_house_value = joblib.load("./data/prep_median_house_value.save")
    return prep_longitude, prep_latitude, prep_median_income, prep_population, prep_median_house_value
