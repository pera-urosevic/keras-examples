import keras


def build_model(hp):
    units = hp.Int("units", min_value=8, max_value=512, step=8)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(units, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def save_model(model):
    model.save("./model/model.keras")


def load_model():
    model = keras.models.load_model("./model/model.keras")
    return model
