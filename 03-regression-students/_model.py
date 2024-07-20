import keras


def build_model(hp):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(units=hp.Int("units_1", min_value=8, max_value=64, step=8), activation="relu"),
            keras.layers.Dense(units=hp.Int("units_2", min_value=8, max_value=64, step=8), activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(0.01))
    return model


def save_model(model):
    model.save("./model/model.keras")


def load_model():
    model = keras.models.load_model("./model/model.keras")
    return model
