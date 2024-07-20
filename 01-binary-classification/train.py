from keras import callbacks
from _data import x_train, y_train, x_test, y_test
from _model import build_model, save_model
from _tuner import build_tuner

tuner = build_tuner()

models = tuner.get_best_models()
best_model = models[0]
best_model.summary()

best_hps = tuner.get_best_hyperparameters(5)
model = build_model(best_hps[0])

tensorboard = callbacks.TensorBoard(log_dir="./logs")
early_stopping = callbacks.EarlyStopping(monitor="accuracy", verbose=1, min_delta=0.01, patience=5)

history = model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[tensorboard, early_stopping])

loss, accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", accuracy)

save_model(model)
