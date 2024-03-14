import keras_tuner
from _model import build_model


def build_tuner(overwrite=False):
    tuner = keras_tuner.Hyperband(
        hypermodel=build_model,
        objective="val_accuracy",
        overwrite=overwrite,
        directory=".",
        project_name="tuner",
    )

    print(tuner.search_space_summary())

    print(tuner.results_summary())
    return tuner
