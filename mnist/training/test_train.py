import numpy as np
from mnist.training.train import train_model, get_model_metrics


def test_train_model():

    b = np.zeros([28, 28, 1], dtype = float)
    b = np.expand_dims(b, axis=0)
    x_train = np.repeat(b, 92)
    x_train = np.reshape(x_train, (92,28, 28, 1))
    y = np.array([0., 1.], dtype=float)
    y_train = np.repeat(y, 92)
    y_train = np.reshape(y_train, (92,2))
    x_test = x_train[0:10]
    y_test = y_train[0:10]
    model = train_model(x_train,y_train,x_test,y_test)
    preds = model.predict(x_test[:1])
    np.testing.assert_almost_equal(preds[0], [0.5,0.5])

def test_get_model_metrics():

    b = np.zeros([28, 28, 1], dtype = float)
    b = np.expand_dims(b, axis=0)
    x_train = np.repeat(b, 92)
    x_train = np.reshape(x_train, (92,28, 28, 1))
    y = np.array([0., 1.], dtype=float)
    y_train = np.repeat(y, 92)
    y_train = np.reshape(y_train, (92,2))
    x_test = x_train[0:10]
    y_test = y_train[0:10]
    model = train_model(x_train,y_train,x_test,y_test)
    metrics = get_model_metrics(model, x_test,y_test)
    val_loss = metrics[0]
    np.testing.assert_almost_equal(val_loss, 0)