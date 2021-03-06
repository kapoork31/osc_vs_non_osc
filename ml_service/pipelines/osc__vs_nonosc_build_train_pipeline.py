from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Workspace, Dataset, Datastore
from azureml.core.runconfig import RunConfiguration
from azureml.train.dnn import TensorFlow
from ml_service.pipelines.load_sample_data import create_sample_data_csv
from ml_service.util.attach_compute import get_compute
from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment
import os


def main():
    e = Env()
    # Get Azure machine learning workspace
    aml_workspace = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group
    )
    print("get_workspace:")
    print(aml_workspace)

    # Get Azure machine learning cluster
    aml_compute = get_compute(
        aml_workspace,
        e.compute_name,
        e.vm_size)
    if aml_compute is not None:
        print("aml_compute:")
        print(aml_compute)

    # Create a reusable Azure ML environment
    environment = get_environment(
        aml_workspace, e.aml_env_name, create_new=e.rebuild_env)
    run_config = RunConfiguration()
    run_config.environment = environment

    if (e.datastore_name):
        datastore_name = e.datastore_name
    else:
        datastore_name = aml_workspace.get_default_datastore().name
    run_config.environment.environment_variables["DATASTORE_NAME"] = datastore_name  # NOQA: E501

    model_name_param = PipelineParameter(
        name="model_name", default_value=e.model_name)
    autoencoder_name_param = PipelineParameter(
        name="autoencoder_name", default_value=e.autoencoder_name)
    dataset_version_param = PipelineParameter(
        name="dataset_version", default_value=e.dataset_version)
    data_file_path_param = PipelineParameter(
        name="data_file_path", default_value="none")
    caller_run_id_param = PipelineParameter(
        name="caller_run_id", default_value="none")

    # Get dataset name
    dataset_name = e.dataset_name
    label_dataset_name = e.label_dataset_name
    # Check to see if dataset exists
    if (dataset_name not in aml_workspace.datasets):
        # This call creates an example CSV from sklearn sample data. If you
        # have already bootstrapped your project, you can comment this line
        # out and use your own CSV.
        create_sample_data_csv()

        # Use a CSV to read in the data set.
        file_name = 'osc__vs_nonosc.csv'

        if (not os.path.exists(file_name)):
            raise Exception("Could not find CSV dataset at \"%s\". If you have bootstrapped your project, you will need to provide a CSV." % file_name)  # NOQA: E501

        # Upload file to default datastore in workspace
        datatstore = Datastore.get(aml_workspace, datastore_name)
        target_path = 'training-data/'
        datatstore.upload_files(
            files=[file_name],
            target_path=target_path,
            overwrite=True,
            show_progress=False)

        # Register dataset
        path_on_datastore = os.path.join(target_path, file_name)
        dataset = Dataset.Tabular.from_delimited_files(
            path=(datatstore, path_on_datastore))
        dataset = dataset.register(
            workspace=aml_workspace,
            name=dataset_name,
            description='osc__vs_nonosc training data',
            tags={'format': 'CSV'},
            create_new_version=True)

    # Create a PipelineData to pass data between steps
    pipeline_data = PipelineData(
        'pipeline_data',
        datastore=aml_workspace.get_default_datastore())

    est = TensorFlow(
        source_directory=e.sources_directory_train,
        entry_script=e.train_script_path,
        compute_target=aml_compute,
        framework_version='2.0',
        pip_packages=['matplotlib',
                      'scikit-learn',
                      'azureml-dataprep[pandas,fuse]']
    )

    train_step = EstimatorStep(
        name="Train Model",
        estimator=est,
        runconfig_pipeline_params=None,
        outputs=[pipeline_data],
        estimator_entry_script_arguments=[
            "--model_name", model_name_param,
            "--autoencoder_name", autoencoder_name_param,
            "--step_output", pipeline_data,
            "--dataset_version", dataset_version_param,
            "--data_file_path", data_file_path_param,
            "--caller_run_id", caller_run_id_param,
            "--dataset_name", dataset_name,
            "--label_dataset_name", label_dataset_name,
            "--n_epochs", e.no_of_epochs,
            "--batch_size", e.batch_size,
            "--autoencoder_n_epochs", e.autoencoder_no_of_epochs,
            "--autoencoder_batch_size", e.autoencoder_batch_size,
        ],
        compute_target=aml_compute,
        allow_reuse=False,
    )

    print("Step Train created")

    evaluate_step = PythonScriptStep(
        name="Evaluate Model ",
        script_name=e.evaluate_script_path,
        compute_target=aml_compute,
        source_directory=e.sources_directory_train,
        arguments=[
            "--model_name", model_name_param,
            "--allow_run_cancel", e.allow_run_cancel,
        ],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Step Evaluate created")

    regest = TensorFlow(
        source_directory=e.sources_directory_train,
        entry_script=e.register_script_path,
        compute_target=aml_compute,
        framework_version='2.0',
    )
    register_step = EstimatorStep(
        name="Register Model",
        estimator=regest,
        runconfig_pipeline_params=None,
        inputs=[pipeline_data],
        estimator_entry_script_arguments=[
            "--model_name", model_name_param,
            "--step_input", pipeline_data,
            "--autoencoder_name", autoencoder_name_param,
        ],
        compute_target=aml_compute,
        allow_reuse=False,
    )

    print("Step Register created")
    # Check run_evaluation flag to include or exclude evaluation step.
    if ((e.run_evaluation).lower() == 'true'):
        print("Include evaluation step before register step.")
        evaluate_step.run_after(train_step)
        register_step.run_after(evaluate_step)
        steps = [train_step, evaluate_step, register_step]
    else:
        print("Exclude evaluation step and directly run register step.")
        register_step.run_after(train_step)
        steps = [train_step, register_step]

    train_pipeline = Pipeline(workspace=aml_workspace, steps=steps)
    train_pipeline._set_experiment_name
    train_pipeline.validate()
    published_pipeline = train_pipeline.publish(
        name=e.pipeline_name,
        description="Model training/retraining pipeline",
        version=e.build_id
    )
    print(f'Published pipeline: {published_pipeline.name}')
    print(f'for build {published_pipeline.version}')


if __name__ == '__main__':
    main()
