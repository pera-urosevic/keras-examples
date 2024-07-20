import keras


def build_model(hp):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(units=hp.Int("units_1", min_value=16, max_value=256, step=16), activation="relu"),
            keras.layers.Dense(units=hp.Int("units_2", min_value=16, max_value=256, step=16), activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(loss="mse", optimizer="adam")
    return model


def save_model(model):
    model.save("./model/model.keras")


def load_model():
    model = keras.models.load_model("./model/model.keras")
    return model
