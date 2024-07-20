from matplotlib.font_manager import json_dump
import numpy as np
import pandas as pd
import json
from keras import layers
from _config import max_features


def train_validate_test_split(df, train_percent=0.6, validate_percent=0.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


data = pd.read_csv("./data/IMDB Dataset.csv")
print(data.head())

words = data["review"].replace("<br />", " ", regex=True)
layer_text_vectorize = layers.TextVectorization(
    max_tokens=max_features,
    output_mode="int",
    standardize="lower_and_strip_punctuation",
)
layer_text_vectorize.adapt(words.values)
vocabulary = layer_text_vectorize.get_vocabulary()
with open("./data/vocabulary.json", "w") as f:
    json.dump(vocabulary, f)
print("vocabulary size", len(vocabulary))


train, validate, test = train_validate_test_split(data, seed=42)
print("train", train.shape)
print("validate", validate.shape)
print("test", test.shape)

train_x = train["review"]
train_y = (train["sentiment"] == "positive").astype(int)
val_x = validate["review"]
val_y = (validate["sentiment"] == "positive").astype(int)
test_x = test["review"]
test_y = (test["sentiment"] == "positive").astype(int)
print(train_x.head())
print(train_y.head())

data.to_csv("./data/train_x.csv", index=False)
train_x.to_csv("./data/train_x.csv", index=False)
train_y.to_csv("./data/train_y.csv", index=False)
val_x.to_csv("./data/val_x.csv", index=False)
val_y.to_csv("./data/val_y.csv", index=False)
test_x.to_csv("./data/test_x.csv", index=False)
test_y.to_csv("./data/test_y.csv", index=False)
