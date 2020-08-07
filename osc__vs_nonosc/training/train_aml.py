"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace
import os
import argparse
from train import train_model, get_model_metrics
from train_data_drift import train_autoencoder, autoencoder_get_model_metrics
import numpy as np
from util.model_helper import get_latest_model
import tensorflow.keras as k
from sklearn.model_selection import train_test_split
from ml_service.util.env_variables import Env

e = Env()


def register_dataset(
    aml_workspace: Workspace,
    dataset_name: str,
    datastore_name: str,
    file_path: str
) -> Dataset:
    datastore = Datastore.get(aml_workspace, datastore_name)
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, file_path))
    dataset = dataset.register(workspace=aml_workspace,
                               name=dataset_name,
                               create_new_version=True)

    return dataset


def main():
    print("Running train_aml.py")

    parser = argparse.ArgumentParser("train")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the Model",
        default="mnist_model.h5",
    )
    parser.add_argument(
        "--autoencoder_name",
        type=str,
        help="Name of the autoencoder Model",
        default="data_drift_model.h5",
    )

    parser.add_argument(
        "--step_output",
        type=str,
        help=("output for passing data to next step")
    )

    parser.add_argument(
        "--dataset_version",
        type=str,
        help=("dataset version")
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        help=("data file path, if specified,\
               a new version of the dataset will be registered")
    )

    parser.add_argument(
        "--caller_run_id",
        type=str,
        help=("caller run id, for example ADF pipeline run id")
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help=("Dataset name. Dataset must be passed by name\
              to always get the desired dataset version\
              rather than the one used while the pipeline creation")
    )

    args = parser.parse_args()

    print("Argument [model_name]: %s" % args.model_name)
    print("Argument [autoencoder_name]: %s" % args.autoencoder_name)
    print("Argument [step_output]: %s" % args.step_output)
    print("Argument [dataset_version]: %s" % args.dataset_version)
    print("Argument [data_file_path]: %s" % args.data_file_path)
    print("Argument [caller_run_id]: %s" % args.caller_run_id)
    print("Argument [dataset_name]: %s" % args.dataset_name)

    model_name = args.model_name
    autoencoder_name = args.autoencoder_name
    step_output_path = args.step_output
    dataset_version = args.dataset_version
    data_file_path = args.data_file_path
    dataset_name = args.dataset_name

    run = Run.get_context()

    exp = run.experiment
    ws = run.experiment.workspace
    tag_name = 'experiment_name'

    autoencoder = get_latest_model(
        autoencoder_name, tag_name, exp.name, ws)

    # Get the dataset
    if (dataset_name):
        if (data_file_path == 'none'):
            dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name, dataset_version)  # NOQA: E402, E501
        else:
            dataset = register_dataset(run.experiment.workspace,
                                       dataset_name,
                                       os.environ.get("DATASTORE_NAME"),
                                       data_file_path)
    else:
        e = ("No dataset provided")
        print(e)
        raise Exception(e)

    # Link dataset to the step run so it is trackable in the UI
    run.input_datasets['training_data'] = dataset
    run.parent.tag("dataset_id", value=dataset.id)

    dataset2 = Dataset.get_by_name(run.experiment.workspace, dataset_name)
    mount_context = dataset2.mount()
    mount_context.start()  # this will mount the file streams
    data = np.load(
        mount_context.mount_point +
        '/image_data_by_person_all4_no_filter_2500_20prc.npy')
    labels = np.load(
        mount_context.mount_point +
        '/labels_by_person_all4_no_filter_2500_20prc.npy')
    mount_context.stop()  # this will unmount the file streams

    labelSubset = labels
    dataSubset = data
    dataSubset = dataSubset.reshape(len(dataSubset), 57, 86, 1)
    labelSubset = k.utils.to_categorical(labelSubset)
    x_train, x_test, y_train, y_test = train_test_split(
                                                        dataSubset,
                                                        labelSubset,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        stratify=labelSubset
                                                        )

    # Train the model
    model = train_model(
        x_train,
        y_train,
        x_test,
        y_test,
        e.no_of_epochs,
        e.batch_size
        )

    # Evaluate and log the metrics returned from the train function
    metrics = get_model_metrics(model, x_test, y_test)
    run.log("test loss", metrics[0])
    run.log("test accuracy", metrics[1])
    run.parent.log("test loss", metrics[0])
    run.parent.log("test accuracy", metrics[1])

    # Pass model file to next step
    os.makedirs(step_output_path, exist_ok=True)
    model_output_path = os.path.join(step_output_path, model_name)
    model.save(model_output_path)

    if (autoencoder is None):

        autoencoder_and_history = train_autoencoder(x_train, x_train)
        autoencoder = autoencoder_and_history[0]
        history = autoencoder_and_history[1]
        test_loss = autoencoder_get_model_metrics(autoencoder, history, x_test)

        run.log('autoencoder training loss', test_loss[0])
        run.log('autoencoder test loss', test_loss[1])
        run.parent.log('autoencoder training loss', test_loss[0])
        run.parent.log('autoencoder test loss', test_loss[1])

        autencoder_output_path = os.path.join(step_output_path,
                                              autoencoder_name
                                              )
        autoencoder.save(autencoder_output_path)
        print('autencoder saved')
    else:
        print('autencoder model already exists')

    # Also upload model file to run outputs for history
    os.makedirs('outputs', exist_ok=True)
    output_path = os.path.join('outputs', model_name)
    model.save(output_path)

    run.tag("run_type", value="train")
    print(f"tags now present for run: {run.tags}")

    run.complete()


if __name__ == '__main__':
    main()
