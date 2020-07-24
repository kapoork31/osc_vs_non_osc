import numpy as np
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D
from tensorflow.keras.models import Model


def train_autoencoder(x_train, x_test):
    input_img = Input(shape=np.shape(x_train[0]))
    # adapt this if using `channels_first` image data format

    x = Conv2D(32, (5, 5), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (15,22, 32)

    x = Conv2D(32, (5, 5), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    # decoded = Cropping2D(cropping=((3, 0), (2, 0)), data_format=None)(x)
    # at this point the representation is (60,88,1)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    # set to 2 cause 2 outputs via softmax.
    # softmax pushes values between 0 and 1. all probs add up to 1.

    # compile model using accuracy to measure model performance

    history = autoencoder.fit(
                              x_train,
                              x_train,
                              epochs=25,
                              batch_size=8,
                              shuffle=True,
                              verbose=0,
                              validation_data=(x_test, x_test)
    )

    return (autoencoder, history)


def autoencoder_get_model_metrics(autoencoder, history, x_test):

    decoded_imgs = autoencoder.predict(x_test)
    x_test_loss = autoencoder.evaluate(x_test, decoded_imgs)
    losses = (history.history['val_loss'][-1], x_test_loss)
    return (losses)
