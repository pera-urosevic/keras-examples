from _data import x_train, y_train, x_test, y_test
from _tuner import build_tuner

tuner = build_tuner(overwrite=True)
tuner.search(
    x_train,
    y_train,
    epochs=50,
    shuffle=True,
    validation_data=(x_test, y_test),
)
