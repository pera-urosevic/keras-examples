import pandas as pd


def get_data():
    dataframe = pd.read_csv("./data/data.csv")
    print(dataframe.shape)
    print(dataframe.head())

    val_dataframe = dataframe.sample(frac=0.2, random_state=42)
    train_dataframe = dataframe.drop(val_dataframe.index)

    print(f"Using {len(train_dataframe)} samples for training and {len(val_dataframe)} for validation")

    x_train = train_dataframe[["x", "y"]]
    y_train = train_dataframe["r"]

    x_test = val_dataframe[["x", "y"]]
    y_test = val_dataframe["r"]

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = get_data()
