import numpy as np
from azureml.core.model import Model
import joblib
import json


def init():
    global model
    model_path = Model.get_model_path(
        model_name="mnist_model.pkl")
    model = joblib.load(model_path)


def run(raw_data, request_headers):
    data = np.array(json.loads(raw_data)['data'])
    pred = model.predict(data)
    y_pred = (pred > 0.5).astype(int)
    return y_pred.argmax(axis=1).tolist()


x_test = np.load('mnist_data/x_test.npy')

init()
test_samples = json.dumps({"data": x_test.tolist()})
test_samples = bytes(test_samples, encoding='utf8')
request_header = {}
prediction = run(test_samples, {})
print("Test result: ", prediction)
