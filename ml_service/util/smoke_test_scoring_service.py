import argparse
import requests
import time
import numpy as np
from azureml.core import Workspace
from azureml.core.webservice import AksWebservice, AciWebservice
from ml_service.util.env_variables import Env
import secrets

b = np.zeros([28, 28, 1], dtype=float)
b = np.expand_dims(b, axis=0)
x_train = np.repeat(b, 2)
input_sample = np.reshape(x_train, (2, 28, 28, 1))
# input = json.dumps({"data": input_sample.tolist()})
input = {"data": input_sample.tolist()}


def call_web_service(e, service_type, service_name):
    aml_workspace = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group
    )
    print("Fetching service")
    headers = {}
    if service_type == "ACI":
        service = AciWebservice(aml_workspace, service_name)
    else:
        service = AksWebservice(aml_workspace, service_name)
    if service.auth_enabled:
        service_keys = service.get_keys()
        headers['Authorization'] = 'Bearer ' + service_keys[0]
    print("Testing service")
    print(". url: %s" % service.scoring_uri)
    output = call_web_app(service.scoring_uri, headers)

    return output


def call_web_app(url, headers):
    print('calling web app')
    print(len(input))
    # Generate an HTTP 'traceparent' distributed tracing header
    # (per the W3C Trace Context proposed specification).
    headers['traceparent'] = "00-{0}-{1}-00".format(
        secrets.token_hex(16), secrets.token_hex(8))

    retries = 600
    for i in range(retries):
        try:
            response = requests.post(
                url, json=input, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if i == retries - 1:
                raise e
            print(e)
            print("Retrying...")
            time.sleep(1)


def main():

    parser = argparse.ArgumentParser("smoke_test_scoring_service.py")

    parser.add_argument(
        "--type",
        type=str,
        choices=["AKS", "ACI", "Webapp"],
        required=True,
        help="type of service"
    )
    parser.add_argument(
        "--service",
        type=str,
        required=True,
        help="Name of the image to test"
    )
    args = parser.parse_args()

    e = Env()
    if args.type == "Webapp":
        output = call_web_app(args.service, {})
    else:
        output = call_web_service(e, args.type, args.service)
    print("Verifying service output")

    print(output)
    assert "result" in output
    print(len(output["result"]))
    # assert len(output["result"]) == output_len
    print("Smoke test successful.")


if __name__ == '__main__':
    main()
