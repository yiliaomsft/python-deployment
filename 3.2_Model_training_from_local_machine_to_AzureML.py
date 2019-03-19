# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Model training on AzureML from same locally run notebook
#
# At the end of this notebook, we will have learned how to:
# - Create an AMLcompute target
# - Trigger the execution of a notebook on AzureML.
#
# We will leverage the 3_Model_training_from_local_machine_to_AzureML.ipynb notebook and the notebook_launcher.py code (both available in this << PUT LINK HERE >> repository). Ultimately we will train a model remotely using the notebook we used to train locally.

# ## Pre-requisites
# It is assumed here that our local environment and Azure subscription are already set up (cf. [details](LINK TO PRIOR notebook)).

# ## Library import

# +
# For automatic reloading of modified libraries
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# Regular python libraries
import matplotlib.pyplot as plt
import os
from sys import platform


# fast.ai
from fastai.vision import *
import torchvision.models as models

# Azure
import azureml.core
from azureml.core import Workspace, Experiment
from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import RunConfiguration
from azureml.core.runconfig import DEFAULT_GPU_IMAGE
from azureml.core.script_run_config import ScriptRunConfig
from azureml.widgets import RunDetails
# -

# ## Azure
#
# ### Workspace
# Let's start by loading the working space which information we saved in our aml_config/config.json file

ws = Workspace.from_config()

# Let's check that the workspace is properly loaded

# Print the workspace attributes
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

# ### Experiment
#
# As we will be running computations on a remote cluster, we need to create an experiment or used one we already have.

# Create a new experiment
experiment_name = 'imgclassif-remote-training'
experiment = Experiment(workspace=ws, name=experiment_name)

print("New experiment:\n --> Name: {}\n --> Workspace name: {}".format(experiment.name, experiment.workspace.name))

# ### Compute resources
#
# We also need to create or retrieve the proper compute resources. Here, we will use an [AMLCompute target](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets) composed of [GPU nodes](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/sizes-gpu).

# +
cluster_name = "gpuclusternc6"

try:
    gpu_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    
    # We are creating here a AMLcompute cluster that leverages GPUs
    # To use CPUs, we should replace "Standard_NC6" by "Standard_D3_v2"
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC6', max_nodes=1)
    # Standard_NC6 machines have only 1 GPU
    
    # We actually create the cluster here
    gpu_cluster = ComputeTarget.create(ws, cluster_name, compute_config)

    # We can also impose a minimum number of nodes and a specific timeout. 
    # If no min_node_count is provided, it uses the scale settings for the cluster
    gpu_cluster.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# We use get_status() to get a detailed status for the current cluster
print(gpu_cluster.get_status().serialize())
# -

# We could also use a "Standard NC12" machine, which has 2 GPUs, and allow for the cluster to scale up to 2 nodes. However, the training of the current model does not require as much power. We are then using only a Standard NC6 machine.

# ### Run configuration
#
# Now that everything is in place, let's create a new run and configure it, so it runs with the libraries we need.

# +
current_directory = os.getcwd()
default_conda_environment = 'myenv2.yml'  # contains import of the papermill library

conda_env = os.path.join(current_directory, default_conda_environment)

# Create a new runconfig object
run_configuration = RunConfiguration()

# Use the GPU cluster we created above
run_configuration.target = cluster_name

# Enable Docker
run_configuration.environment.docker.enabled = True

# Set Docker base image to the default GPU-based image
run_configuration.environment.docker.base_image = DEFAULT_GPU_IMAGE

# Use our myenv.yml file to create a conda environment in the Docker image for execution
# Here, user_managed_dependencies is False because the Docker image does not have the libraries installed yet
run_configuration.environment.python.user_managed_dependencies = False

# Auto-prepare the Docker image when used for execution (if it is not already prepared)
run_configuration.auto_prepare_environment = True

# Specify CondaDependencies object, add necessary packages
# Here, we use our myenv.yml file, as i contains all the packages we need
run_configuration.environment.python.conda_dependencies = CondaDependencies(conda_dependencies_file_path=conda_env)
# -

script_run_config = ScriptRunConfig(source_directory='./from_local_to_remote/',
                                    script='notebook_launcher.py', 
                                    arguments=['--notebook_path', 
                                               '3_Model_training_from_local_machine_to_AzureML.ipynb'],
                                    run_config=run_configuration)

# ### Run execution

run = experiment.submit(script_run_config)

# From here, we can track and visualize the progress of training.

run

# ### Progress and results visualization

print(run.get_portal_url())
run.wait_for_completion(show_output=True)

# <img src="screenshots/80_driver_log.jpg" width="500" align="right">
#
# The `run.wait_for_completion(show_output=True)` command displays the content of the logs files that are stored on the Azure portal. These logs can be found by:
# - Clicking on "Experiments"
# - Selecting the name of the experiment considered here
# - Clicking on the run number currently active
# - Clicking on the "Logs" tab

# Note: This run involves several steps:
# 1. Preparation of the AMLCompute target and import of all necessary libraries
# 2. Queuing of the job to run, i.e. resizing of the cluster from 0 to 1 node
# 3. Actual execution of the job
#
# The first step can take up to 20 min. Once the preparation phase has completed, steps 2. and 3. can be run again. This is especially convenient when we want to make a change to either the notebook to be run remotely, or to the notebook_launcher.py script.

# Details on the nodes status can also be seen directly in the notebook by running the `RunDetails(run).show()` command.

RunDetails(run).show()

# Once the run is completed, we can retrieve the list of files that were generated:
# - Log files
# - Model and output notebook we stored in the "outputs/" folder.
#
# The output notebook is the same as the original 3_Model_training_from_local_machine_to_AzureML.ipynb, but also contains the results of each cell, as executed on the AMLCompute target.

run.get_file_names()

# The use of the `papermill` library, that allowed us to run the 3_Model_training_from_local_machine_to_AzureML.ipynb notebook without any changes, on the AMLCompute target, also allowed us to retrieve metrics and plot them both on the Azure portal and in this current notebook.
#
# On the portal, these can be accessed under "Experiments", and after clicking on the experiment and then run number of interest.
#
# <img src="screenshots/remote_training_metrics.jpg" width="800" align="left">

# In this notebook, we need to retrieve the data though the `run` object.

print(run.get_metrics().keys())
for k in run.get_metrics().keys():
    print("{}: {} values".format(k, len(run.get_metrics()[k])))

training_loss = run.get_metrics()['training_loss']
accuracy = [x for x in run.get_metrics()['accuracy']]
learning_rate = run.get_metrics()['learning_rate']
validation_loss = run.get_metrics()['validation_loss']

# +
fig, ax = plt.subplots(2,2, figsize=(8,8))
ax[0, 0].plot(training_loss)
ax[0, 0].set_title('training loss')

ax[1, 0].plot(learning_rate)
ax[1, 0].set_title('learning rate')

ax[0, 1].plot(accuracy)
ax[0, 1].set_title('accuracy')

ax[1, 1].plot(validation_loss)
ax[1, 1].set_title('validation loss')

# +
# Run the following in the terminal first:

# jupyter nbextension uninstall --py --user azureml.widgets
# jupyter nbextension uninstall --py --user azureml.train.widgets

# +
# Reference notebook: https://github.com/Microsoft/Recommenders/blob/danielsc/azureml/notebooks/02_train_in_the_cloud/train_fastai_on_aml.ipynb
# Papermill: https://github.com/nteract/papermill
