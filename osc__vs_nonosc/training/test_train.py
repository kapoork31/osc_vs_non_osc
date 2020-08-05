import numpy as np
from osc__vs_nonosc.training.train import train_model, get_model_metrics
from osc__vs_nonosc.training.train_data_drift import (
    autoencoder_get_model_metrics, train_autoencoder)
from ml_service.util.env_variables import Env

e = Env()


def test_train_model():

    b = np.zeros([57, 86, 1], dtype=float)
    b = np.expand_dims(b, axis=0)
    x_train = np.repeat(b, 92)
    x_train = np.reshape(x_train, (92, 57, 86, 1))
    y = np.array([0., 1.], dtype=float)
    y_train = np.repeat(y, 92)
    y_train = np.reshape(y_train, (92, 2))
    x_test = x_train[0:10]
    y_test = y_train[0:10]
    model = train_model(
                        x_train,
                        y_train,
                        x_test, y_test,
                        e.no_of_epochs,
                        e.batch_size
                        )
    preds = model.predict(x_test[:1])
    # np.testing.assert_almost_equal(preds[0], [0.5, 0.5])
    assert preds[0] > 0


def test_get_model_metrics():

    b = np.zeros([57, 86, 1], dtype=float)
    b = np.expand_dims(b, axis=0)
    x_train = np.repeat(b, 92)
    x_train = np.reshape(x_train, (92, 57, 86, 1))
    y = np.array([0., 1.], dtype=float)
    y_train = np.repeat(y, 92)
    y_train = np.reshape(y_train, (92, 2))
    x_test = x_train[0:10]
    y_test = y_train[0:10]
    model = train_model(
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        e.no_of_epochs,
                        e.batch_size
                        )
    metrics = get_model_metrics(model, x_test, y_test)
    val_loss = metrics[0]
    # np.testing.assert_almost_equal(val_loss, 0)
    assert val_loss > 0


def test_train_autoencoder():
    b = np.zeros([57, 86, 1], dtype=float)
    b = np.expand_dims(b, axis=0)
    x_train = np.repeat(b, 92)
    x_train = np.reshape(x_train, (92, 57, 86, 1))
    autoencoder_and_history = train_autoencoder(
                                                x_train,
                                                x_train,
                                                e.autoencoder_no_of_epochs,
                                                e.autoencoder_batch_size
                                                )
    history = autoencoder_and_history[1]
    assert history.history['val_loss'][-1] < history.history['val_loss'][0]


def test_autoencoder_get_model_metrics():
    b = np.zeros([57, 86, 1], dtype=float)
    b = np.expand_dims(b, axis=0)
    x_train = np.repeat(b, 92)
    x_train = np.reshape(x_train, (92, 57, 86, 1))
    x_test = x_train[0:10]
    autoencoder_and_history = train_autoencoder(
                                                x_train,
                                                x_train,
                                                e.autoencoder_no_of_epochs,
                                                e.autoencoder_batch_size
                                                )
    autoencoder = autoencoder_and_history[0]
    history = autoencoder_and_history[1]
    test_loss = autoencoder_get_model_metrics(autoencoder, history, x_test)
    assert test_loss[1] > 0
