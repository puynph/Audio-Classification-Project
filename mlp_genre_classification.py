import json
import numpy as np
from sklearn.model_selection import train_test_split
import keras


DATASET_PATH = "data.json"


def load_data(dataset):
    """
    Load dataset from json file
    :return X (nparray): Inputs
    :return y (nparray): Targets
    """

    with open (dataset, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["label"])

    print("Data is successfully loaded!")

    return X, y


if __name__ == "__main__":
    # load data
    X, y = load_data(DATASET_PATH)
    y = y - 1  # shift the label

    # train/test set split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build network model
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # hidden layers
        keras.layers.Dense(1024, activation="relu"),

        keras.layers.Dense(512, activation="relu"),

        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.5),  # Adjust dropout rate

        # output layer
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # train model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)

