from azureml.core import Workspace
from azureml.core.runconfig import RunConfiguration
from ml_service.util.env_variables import Env
from ml_service.util.attach_compute import get_compute
from azureml.pipeline.core import PublishedPipeline
# from azureml.pipeline.steps import PythonScriptStep
from ml_service.util.manage_environment import get_environment
from azureml.data.data_reference import DataReference


def main():
    # run = Run.get_context()
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
        e.vm_size
    )

    if aml_compute is not None:
        print("aml_compute:")
        print(aml_compute)

    pipeline_name = e.scoring_pipeline_name
    pub_pipelines = PublishedPipeline.list(
        aml_workspace,
        active_only=True,
        _service_endpoint=None
    )
    pipeline_exists = False

    for p in pub_pipelines:
        if(p.name == pipeline_name):
            pipeline_exists = True

    if(pipeline_exists):

        environment = get_environment(
            aml_workspace, e.aml_env_name, create_new=e.rebuild_env)
        run_config = RunConfiguration()
        run_config.environment = environment

        if (e.datastore_name):
            datastore_name = e.datastore_name
        else:
            datastore_name = aml_workspace.get_default_datastore().name

        run_config.environment.environment_variables[
            "DATASTORE_NAME"
        ] = datastore_name  # NOQA: E501

        def_blob_store = aml_workspace.get_default_datastore()

        output_dir = DataReference(
            datastore=def_blob_store,
            data_reference_name="output_data",
            path_on_datastore=e.scoring_script_output_path)

        input_dir_raw = DataReference(
            datastore=def_blob_store,
            data_reference_name="input_data_raw",
            path_on_datastore=e.scoring_script_input_raw)

        input_dir_meta = DataReference(
            datastore=def_blob_store,
            data_reference_name="input_data_meta",
            path_on_datastore=e.scoring_script_input_meta)

        print(output_dir)
        print(input_dir_raw)
        print(input_dir_meta)


if __name__ == '__main__':
    main()
