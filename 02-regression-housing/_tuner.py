import keras_tuner
from _model import build_model


def build_tuner(overwrite=False):
    tuner = keras_tuner.Hyperband(
        hypermodel=build_model,
        max_epochs=20,
        objective="val_loss",
        project_name="tuner",
        overwrite=overwrite,
        directory=".",
    )
    print(tuner.search_space_summary())
    return tuner
