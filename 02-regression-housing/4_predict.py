from matplotlib import pyplot as plt
import pandas as pd
from _data import get_preprocessors
from _model import load_model

prep_longitude, prep_latitude, prep_median_income, prep_population, prep_median_house_value = get_preprocessors()


def predict(model, lon, lat, inc, pop, val):
    x = pd.DataFrame([[lon, lat, inc, pop]], columns=["longitude", "latitude", "median_income", "population"])
    x["longitude"] = prep_longitude.transform(x.loc[:, ["longitude"]])
    x["latitude"] = prep_latitude.transform(x.loc[:, ["latitude"]])
    x["median_income"] = prep_median_income.transform(x.loc[:, ["median_income"]])
    x["population"] = prep_population.transform(x.loc[:, ["population"]])
    prediction = model.predict(x, verbose=0)
    y = prep_median_house_value.inverse_transform(prediction)[0][0]
    return val, y


model = load_model()
res = pd.DataFrame(columns=["real", "predicted"])
data = pd.read_csv("./data/california_housing.csv")
samples = data.sample(n=1000, random_state=42)
for index, row in samples.iterrows():
    res.loc[index] = predict(model, row["longitude"], row["latitude"], row["median_income"], row["population"],
                             row["median_house_value"])
res.plot.scatter(x="real", y="predicted")
plt.show()
