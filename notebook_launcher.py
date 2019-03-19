import argparse
from azureml.core import Run
from collections import defaultdict
from datetime import datetime
import numpy as np
import os
import papermill as pm
import sys
from sys import platform

os.system(sys.executable + ' -m ipykernel install --user --name ipyazureml')


def execute_notebook(notebook_path, parameters):
    print("2. Executing the notebook - {}".format(datetime.now()))
    pm.execute_notebook(
        notebook_path,
        'outputs/output.ipynb',
        kernel_name='ipyazureml',
        parameters=parameters,
    )
    print("3. Retrieving the output - {}".format(datetime.now()))
    results = pm.read_notebook('outputs/output.ipynb').dataframe.set_index("name")["value"]

    print("4. Extracting the values - {}".format(datetime.now()))
    run = Run.get_context()

    for key, value in results.items():
        if isinstance(value, list):
            if len(value) > 50:
                # 50 = max number of values allowed per log_list call
                # This is not to go over the 3kB limit
                # If size(value) > 50, break down value in several chunks
                value_size = len(value)
                number_chunks = int(value_size / 50)
                if value_size % 50 != 0:
                    number_chunks += 1

                for k in range(0, number_chunks):
                    start_index = k * 50
                    end_index = np.min([start_index + 50, value_size])
                    subset = value[start_index:end_index]

                    run.log_list(key, subset)

            else:
                run.log_list(key, value)
        else:
            run.log(key, value)

    return results


if __name__ == '__main__':
    print("We are running on a {} OS.".format(platform))
    print("1. Parsing the arguments - {}".format(datetime.now()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook_path')
    parser.add_argument('--parameters')
    FLAGS, unparsed = parser.parse_known_args()

    parameters = {unparsed[i]: unparsed[i+1] for i in range(0, len(unparsed), 2)}
    print(FLAGS.notebook_path, parameters)
    execute_notebook(notebook_path = FLAGS.notebook_path, 
                     parameters=parameters)
