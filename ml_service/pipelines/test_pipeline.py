import argparse
import sys
import os
from azureml.core import Run, Experiment, Workspace
from ml_service.util.env_variables import Env
from mnist.util.model_helper import get_latest_model
from azureml.pipeline.core import PublishedPipeline

def main():
    print('hello')

    run = Run.get_context()
    print(run)

if __name__ == '__main__':
    main()
