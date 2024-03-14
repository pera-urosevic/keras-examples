import numpy as np
from _model import load_model


def predict(model, px, py):
    tr = abs(px + py) < 5
    new_data = np.array([[px, py]])
    prediction = model.predict(new_data, verbose=0)
    r = bool(round(prediction[0][0]))
    s = "✔️" if r == tr else "❌"
    print(f"{s}  for ({px}, {py}) = {r}, actual = {tr}")


model = load_model()
predict(model, 3, -2)
predict(model, 5, 5)
predict(model, 8, -8)
predict(model, 8, 8)
