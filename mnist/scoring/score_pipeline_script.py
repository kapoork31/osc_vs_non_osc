from azureml.core import Run
from azureml.core.model import Model
import numpy as np
import os
from tensorflow.keras.models import load_model
import argparse
import pandas as pd


def most_frequent(List):
    return max(set(List), key=List.count)


parser = argparse.ArgumentParser("predict")
parser.add_argument(
    "--model_name",
    type=str,
    help="Name of the Model",
    default="mnist_model.h5",
)
parser.add_argument(
    "--model_name_autoencoder",
    type=str,
    help="Name of the Model",
    default="data_drift_model.h5",
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="output_dir"
)  # read in output dir
parser.add_argument(
    "--input_dir_raw",
    type=str,
    help="input_dir_raw"
)  # read in input raw dir
parser.add_argument(
    "--input_dir_meta",
    type=str,
    help="input_dir_meta"
)  # read in input meta dir

args = parser.parse_args()  # get arg values and store in variable
output_dir = args.output_dir
input_dir_meta = args.input_dir_meta
input_dir_raw = args.input_dir_raw

run = Run.get_context()

if(os.path.exists(input_dir_meta) and
        os.path.exists(input_dir_raw)):  # if both input dirs exist

    # print("Argument [model_name]: %s" % args.model_name)
    model_name = args.model_name
    autoencoder_name = args.model_name_autoencoder

    file_meta_processed_path = output_dir + '/' + 'file_processed.csv'
    # filepath of processed filenames
    file_processed_exists = os.path.exists(file_meta_processed_path)
    # boolean if processed filenames file exists

    allFiles = os.listdir(input_dir_raw)
    # all files in input raw dir
    npy_files = [s for s in allFiles if "npy" in s]
    # all npy files in the input directory

    if(len(npy_files) > 0):  # if there is minimumm 1 npy file

        npy = npy_files[0]  # file name
        to_execute = True  # set to_execute to true

        if(file_processed_exists):  # if the filename meta exists
            file_meta = pd.read_csv(file_meta_processed_path)
            # read the processed filename
            files_processed = file_meta.filename.unique()
            # names of the processed files
            if(npy in files_processed):
                # if the npy files has been processed then dont execute
                to_execute = False
                print('file already processed')

        if(to_execute):  # then execute

            meta_data = pd.read_csv(input_dir_meta + '/' +
                                    'test_data_to_predict_meta.csv'
                                    )
            data_to_predict = np.load(input_dir_raw + '/' + npy)
            width = np.shape(data_to_predict[0])[1]
            height = np.shape(data_to_predict[0])[0]
            print(np.shape(data_to_predict))
            data_to_predict = data_to_predict.reshape(
                len(data_to_predict),
                height,
                width,
                1
            )

            ws = Run.get_context().experiment.workspace

            model_root = Model.get_model_path(
                model_name,
                version=None,
                _workspace=ws
            )
            model = load_model(model_root)

            autoencoder_root = Model.get_model_path(
                autoencoder_name,
                version=None,
                _workspace=ws
            )
            autoencoder = load_model(autoencoder_root)
            decoded_imgs = autoencoder.predict(data_to_predict)
            x_test_loss = autoencoder.evaluate(data_to_predict, decoded_imgs)
            run.log('autoencoder test loss', x_test_loss)
            run.parent.log('autoencoder test loss', x_test_loss)

            meta_df = pd.DataFrame()
            unique_sessions = meta_data.sessionId.unique()
            for us in unique_sessions:
                temp = meta_data[meta_data.sessionId == us]
                first = temp.first_index.tolist()[0]
                last = temp.last_index.tolist()[0] + 1
                data_to_predict_temp = data_to_predict[first:last]
                pred = model.predict(data_to_predict_temp)
                y_pred = (pred > 0.5).astype(int)
                res = y_pred.argmax(axis=1).tolist()
                sessionDevice = most_frequent(res)
                devices_data = {'device': [sessionDevice], 'sessionId': [us]}
                df = pd.DataFrame(
                    devices_data,
                    columns=['device', 'sessionId']
                )
                meta_df = meta_df.append(df)

            os.remove(input_dir_raw + '\\' + npy)
            # deleted the npy file
            print('deleted npy data')
            file_meta_dict = {'filename': [npy]}
            file_meta_df = pd.DataFrame(file_meta_dict, columns=['filename'])

            meta_processed_path = output_dir + '\\' + 'processed.csv'
            processed_exists = os.path.exists(meta_processed_path)
            # if the meta processed file exists

            if(processed_exists):  # if meta_processed_path exists then append
                meta_df.to_csv(
                    meta_processed_path,
                    mode='a',
                    index=False,
                    header=False
                )
            else:  # else than create
                meta_df.to_csv(meta_processed_path, mode='w', index=False)

            if(file_processed_exists):  # if processed file exists then append
                file_meta_df.to_csv(
                    file_meta_processed_path,
                    mode='a',
                    index=False,
                    header=False
                )
            else:  # else then create
                file_meta_df.to_csv(
                    file_meta_processed_path,
                    mode='w',
                    index=False
                )

    else:
        print('no data to predict exists')
else:
    print('no input folder exists')

run.complete()
