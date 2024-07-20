from keras import callbacks
from _data import get_train_data, get_test_data
from _model import build_model, save_model
from _tuner import build_tuner

tuner = build_tuner()

models = tuner.get_best_models()
best_model = models[0]
best_model.summary()

best_hps = tuner.get_best_hyperparameters(5)
model = build_model(best_hps[0])

tensorboard = callbacks.TensorBoard(log_dir="./logs")
early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1)

train_x, train_y = get_train_data()
history = model.fit(train_x, train_y, epochs=20, batch_size=32, callbacks=[tensorboard, early_stopping])

test_x, test_y = get_test_data()
ta = model.evaluate(test_x, test_y)
print("Test accuracy:", ta)

save_model(model)
