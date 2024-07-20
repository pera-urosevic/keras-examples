import keras
from _data import get_train_data, get_test_data
from _tuner import build_tuner

train_x, train_y = get_train_data()
test_x, test_y = get_test_data()

tuner = build_tuner(overwrite=True)
tuner.search(
    train_x,
    train_y,
    epochs=20,
    shuffle=True,
    validation_data=(test_x, test_y),
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)],
)
