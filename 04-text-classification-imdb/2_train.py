import json
from keras import models, layers, callbacks
import pandas as pd
import tensorflow as tf
from _config import max_features, embedding_dim

vocabulary = json.load(open("./data/vocabulary.json"))
train_x = pd.read_csv("./data/train_x.csv")
train_y = pd.read_csv("./data/train_y.csv")
val_x = pd.read_csv("./data/val_x.csv")
val_y = pd.read_csv("./data/val_y.csv")
test_x = pd.read_csv("./data/test_x.csv")
test_y = pd.read_csv("./data/test_y.csv")
print("vocabulary size", len(vocabulary))

model = models.Sequential(
    [
        layers.Input(shape=(1,), dtype=tf.string),
        layers.TextVectorization(vocabulary=vocabulary),
        layers.Embedding(max_features + 1, embedding_dim),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

early_stopping = callbacks.EarlyStopping(monitor="accuracy", min_delta=0.01, verbose=1)
model.fit(train_x, train_y, epochs=10, validation_data=(val_x, val_y), batch_size=32, callbacks=[early_stopping])
print(model.summary())

loss, accuracy = model.evaluate(test_x, test_y)
print("loss", loss)
print("accuracy", accuracy)

model.save("./model/model.keras")
