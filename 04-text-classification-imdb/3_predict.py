import tensorflow as tf
from keras import models


def predict_sentiment(text):
    example = tf.constant([text])
    prediction = model.predict(example, verbose=0)
    p = "👍" if prediction[0][0] > 0.5 else "👎"
    print(f"{p} {text}")


model = models.load_model("./model/model.keras")

print('👍 = positive sentiment, 👎 = negative sentiment\n')

predict_sentiment("The movie was great!")
predict_sentiment("The movie was okay.")
predict_sentiment("The movie was terrible...")

predict_sentiment("awful")
predict_sentiment("okay")
predict_sentiment("best")

predict_sentiment("i enjoyed it")
predict_sentiment("boring and absurd")
predict_sentiment("most of the time good, but bad in some parts")
