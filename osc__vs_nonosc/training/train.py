import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization)


def train_model(x_train, y_train, x_test, y_test, n_epochs, batch_size):
    model = Sequential()
    # add model layers
    model.add(Conv2D(
                    32,
                    kernel_size=(9, 9),
                    activation='relu',
                    input_shape=(57, 86, 1)
                    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='softmax'))
    # set to 2 cause 2 outputs via softmax.
    # set to 2 cause 2 outputs via softmax.
    # softmax pushes values between 0 and 1. all probs add up to 1.

    # compile model using accuracy to measure model performance
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=n_epochs,
        verbose=0
    )
    return (model, hist)


def get_model_metrics(model, x_test, y_test):
    results = model.evaluate(x_test, y_test)
    return results


def main():
    # Load Data
    x_train = np.load('mnist_data/x_train.npy')
    y_train = np.load('mnist_data/y_train.npy')
    x_test = np.load('mnist_data/x_test.npy')
    y_test = np.load('mnist_data/y_test.npy')

    model = train_model(x_train, y_train, x_test, y_test)[0]

    # Validate Model on Validation Set
    metrics = get_model_metrics(model, x_test, y_test)
    print(metrics)
    # Save Model
    model_name = "mnist_model.h5"
    model.save(model_name)


if __name__ == '__main__':
    main()
