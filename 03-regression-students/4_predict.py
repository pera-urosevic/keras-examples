from matplotlib import pyplot as plt
import pandas as pd
from _data import get_preprocessors
from _model import load_model

preps = get_preprocessors()


def predict(model, row):
    vals = [
        [
            row["Hours Studied"],
            row["Sleep Hours"],
            row["Previous Scores"],
            row["Sample Question Papers Practiced"],
        ]
    ]
    columns = ["Hours Studied", "Sleep Hours", "Previous Scores", "Sample Question Papers Practiced"]
    x = pd.DataFrame(vals, columns=columns)
    x["Hours Studied"] = preps["Hours Studied"].transform(x.loc[:, ["Hours Studied"]])
    x["Sleep Hours"] = preps["Sleep Hours"].transform(x.loc[:, ["Sleep Hours"]])
    x["Previous Scores"] = preps["Previous Scores"].transform(x.loc[:, ["Previous Scores"]])
    x["Sample Question Papers Practiced"] = preps["Sample Question Papers Practiced"].transform(
        x.loc[:, ["Sample Question Papers Practiced"]]
    )
    prediction = model.predict(x, verbose=0)
    y = preps["Performance Index"].inverse_transform(prediction)[0][0]
    return row["Performance Index"], y


n = 1000
model = load_model()
res = pd.DataFrame(columns=["pred", "real"])
data = pd.read_csv("./data/Student_Performance.csv").dropna()
samples = data.sample(n=n, random_state=42)
correct = 0
for index, row in samples.iterrows():
    real, pred = predict(model, row)
    if real == pred:
        correct += 1
    res.loc[index] = pred, real
print(res.head(30))
res.plot.scatter(x="real", y="pred")
plt.show()
