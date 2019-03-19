# -*- coding: utf-8 -*-
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

# ## Deployment of a model as a service with Azure Container Instances
#
# At the end of this notebook, we will have learned how to:
# - Train an image classifier model locally using [fast.ai](https://docs.fast.ai/)
# - Deploy this model as a web service using an [Azure Container Instance](https://azure.microsoft.com/en-us/services/container-instances/)
# - Test that the service works well when called for the notebook
#
#
# <img src="screenshots/ACI_diagram.png" width="500" align=left>

# ### Pre-requisite
#
# For this notebook to run properly on your machine, the following should already be in place:

# #### Local machine setup
# - Miniconda virtual environment - Please follow [these instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python), if you do not have one yet
# - Python 3.6 (Tensorboard and some Azure elements are not supported with Python 3.7)
# - Azure SDK: [Steps](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python#install-the-sdk-1) to follow to install it
# - Pytorch: `conda install –c pytorch pytorch-cpu torchvision`
# - Fastai: `conda install fastai=1.0.44 -c fastai –c conda-forge`
#   - Please follow [these instructions](https://github.com/fastai/fastai/blob/master/README.md#installation), if you encounter any issues with the above

# #### Azure subscription setup
# You also need an account on the Azure platform. If you do not have one:
# - [Create an account](https://azure.microsoft.com/en-us/free/services/machine-learning/)
# - [Add a subscription](https://ms.portal.azure.com/#blade/Microsoft_Azure_Billing/SubscriptionsBlade) -- You can start with a free one
# - [Add a resource group](https://ms.portal.azure.com/#blade/HubsExtension/Resources/resourceType/Microsoft.Resources%2Fsubscriptions%2FresourceGroups)
#
# What is required here is:
# - A subscription ID
# - A resource group

# ### Library import
# Throughout this notebook, we will be using a variety of libraries. We are listing them here for better readibility.

# +
# For automatic reloading of modified libraries
# %reload_ext autoreload
# %autoreload 2

# Regular python libraries
from dotenv import load_dotenv, find_dotenv
import json
import numpy as np
import os
import requests
from shutil import copy2
from string import digits
from sys import platform
from utils import *
import webbrowser

# fast.ai
from fastai.vision import *
import torchvision.models as models

# Azure
import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.telemetry import set_diagnostics_collection
from azureml.contrib.tensorboard import Tensorboard
from azureml.contrib.tensorboard.export import export_to_tensorboard
from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice, Webservice
# -

# ### Azure workspace

# #### SDK version
#
# Before we start, let's check which version of the Azure SDK we are working with.

# Check core SDK version number
print("Azure ML SDK Version: {}".format(azureml.core.VERSION))

# #### Workspace creation
# Now that we have our environment and proper libraries in place, let's load an existing workspace or create a new one on our Azure account, and save it to a local configuration file. For this, we need our:
# - subscription ID
# - resource group
# - location
# - name of the workspace we want to create (e.g. 'myworkspace').
#
# Not to write our credentials in plain text, we can use a the [python-dotenv](https://github.com/theskumar/python-dotenv/blob/master/README.md) library and a [`.env`](https://drivendata.github.io/cookiecutter-data-science/#keep-secrets-and-configuration-out-of-version-control) file. This file contains all our credentials and never gets pushed to git (as long as it is listed in our `.gitignore` file), if we use source control. To load our credentials into our notebook, we simply use:
# - `find_dotenv()`: to find the `.env` file by walking up our directory hierarchy
# - `load_dotenv()`: to load the entries as environment variables
#
# In our case, the `.env` file contains the values for the 4 variables listed below.

# +
# Extraction of the variable values
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Assignment of these values to our workspace variables
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
workspace_region = os.getenv("WORKSPACE_REGION")
# -

try:
    # Load the workspace
    if os.path.exists('./aml_config/config.json'):
        # From a configuration file
        ws = Workspace.from_config()
    else:
        # Or directly from Azure
        ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
        # and generate a local configuration file
        ws.write_config()
    print("Workspace configuration succeeded.")
except:
    # Create a workspace from scratch
    ws = Workspace.create(name=workspace_name,
                          subscription_id=subscription_id,
                          resource_group=resource_group,
                          create_resource_group=True,
                          location=workspace_region
                         )
    print("Workspace {} is not accessible. Change your parameters or create a new workspace.")

# Let's check that the workspace is properly loaded

# Print the workspace attributes
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

# We can see this workspace on the Azure portal by sequentially clicking on:
# - Resource groups, and clicking the one we referenced above

# <img src="screenshots/resource_group.jpg" width="800">

# - workspace_name

# <img src="screenshots/workspace.jpg" width="800">

# #### Experiment

# An experiment contains a series of trials called `Runs`. In the present case, the `run` will consist in the training of our model. Through this `run`'s methods, we will be able to log several metrics such as training and test loss and accuracy, and to tag our run. The full decsription of the `run` class is available [here](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py).

# Create a new experiment
experiment_name = 'image-classifier-webservice'
experiment = Experiment(workspace=ws, name=experiment_name)

print("New experiment:\n --> Name: {}\n --> Workspace name: {}".format(experiment.name, experiment.workspace.name))

# While our `experiment` object is now created, it is not visible on the Azure portal yet.

# #### Diagnostics <<< See what that actually does, i.e. what is being monitored
#
# >>>> Add blurb about what gets tracked and how to get access to it <L<<

set_diagnostics_collection(send_diagnostics=True)

# #### Run initialization

# Initialize the run
runner = experiment.start_logging()

# Now that we have run an action against our experiment object, we can see the latter on the Azure portal, under `Experiments` (in the left-hand side list).
#
# <img src="screenshots/experiment.jpg" width="800">

# From here, we will be able to see a visual representation, on the Azure portal, of metrics tracked during training. For that, let's click on:
# - *image-classifier-with-fastai*, under `Experiments`
# - the run number at the top of the displayed list
#
# <img src="screenshots/experiment_runs_list.jpg" width="800">

# ### Data retrieval and pre-processing

# The data we are using here are structured as follows:
# ```
# dataset/
#    class1/
#    class2/
#    ...
# ```
# Fast.ai's [ImageItemList](https://docs.fast.ai/vision.data.html#ImageItemList) class allows us to read them as they are, and to create a `train` and a `valid` sets.
#
# Caution: As fast.ai processes the data as batches, it is important to adjust the batch size to the size of our training dataset. By default, fast.ai uses a batch size of 64 images. If the total size of our training set is smaller than that, we need to reduce the batch size to our training set size, at most. Thus, with a training set of 44 images, we need to change our batch size to a value <= 44. If we do not do that, we are likely to not get our data loaded in the ImageDataBunch object and to see the following error: <img src="screenshots/imagedatabunch_batchsize_error.jpg" width="500">

# Knowing this, we can now:
# 1. Set the percentage and random seed to consider for the training/validation split
# 2. Count the number of files we will have in the training set to determine the the proper batch size
# 3. Create the ImageDataBunch object through the `ImageItemList` class

# Path to our data
# imgs_path = '/Users/alteste/Repos/ComputerVisionBestPractices/image_classification/data/imgs_recycling'
imgs_path = '../../data/imgs_recycling'

file_count = 0
for root, dirs, files in os.walk(imgs_path):
    file_count += len(files)
print("Our full dataset contains {} images".format(file_count))

random_seed = 456
validation_percentage = 0.2

# +
train_file_count = int(file_count*(1-validation_percentage))

new_batch_size = int(np.min([train_file_count, 64]))

print("Our training dataset will contain {} images and our batch size be {}".format(file_count, new_batch_size))

# +
data = (ImageItemList.from_folder(imgs_path) # Data location: in path and its subfolders
        .random_split_by_pct(valid_pct=validation_percentage,
                             seed=random_seed) # Train/valid split: with percentage of all data
        .label_from_re(pat="msft-(\D+)\d+.jpg", full_path=False) # Label: extracted from filenames
        .transform() # Data augmentation: none here
        .databunch(bs=new_batch_size)) # Conversion to ImageDataBunch: with custom batch size

# data3 = (ItemList.from_folder(imgs_path) # Data location: in path and its subfolders
#         .random_split_by_pct(valid_pct=validation_percentage,
#                              seed=random_seed) # Train/valid split: with percentage of all data
#         .label_from_re(pat="msft-(\D+)\d+.jpg", full_path=False) # Label: extracted from filenames
#         .transform() # Data augmentation: none here
#         .databunch(bs=40)) # Conversion to ImageDataBunch: with custom batch size
# -

data.train_ds.x

# Let's make sure that our data were ingested properly by:
# 1. Visualizing a few of our images

data.show_batch(rows=3, figsize=(8,8))

# 2. Listing the classes present in the data

data.classes

# 3. Checking that we have the expected number of training and validation images

print("Training dataset: {} samples\nTesting dataset: {} samples"
      .format(len(data.train_ds.x), len(data.valid_ds.x)))

# ### Model training (locally)

# Now that our data are ready, let's train our model. As the main focus of this notebook is deployment as a web service, we consider here a pre-trained model. We freeze all its layers, except the last one that we will train with our own dataset. We chose AlexNet here, but could have chosen any other model among ResNet, DenseNet, SqueezeNet and VGG. Our model will be optimized on accuracy (which is fine as long as our dataset is balanced). For this, we use fast.ai's [create_cnn](https://docs.fast.ai/vision.learner.html#create_cnn) function.
#
# Note: If you are working on a Windows machine, depending on your environment setup, you may observe that `create_cnn` hangs and never completes. If that is the case, use the `custom_create_cnn()` function instead. It is part of the `utils.py` file that accompanies this notebook. To use it, just run:
#
# `import utils
# learn = custom_create_cnn(data, mdl_object, metrics=error_rate)`

# +
mdl_object = models.resnet18 # any model among alexnet, resnet18/34/50/101/152, squeezenet1.0/1.1, densenet121/161/169/201, vgg16/19

if platform == 'win32':
    learn = custom_create_cnn(data, mdl_object, metrics=error_rate)
else:
    learn = create_cnn(data, mdl_object, metrics=error_rate)
# -

# We train our model over 5 epochs and over one [learning rate cycle](https://medium.com/@nachiket.tanksale/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6). **This may take a few minutes.**

learn.fit_one_cycle(5)

# Let's now look at how our model actually performed against the validation set, and record the training and validation losses.

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# Fast.ai also allows us to identify the worst predictions, i.e. top losses

interp.plot_top_losses(4, figsize=(8,8))

interp.most_confused()

# ### Model export
#
# To be able to use the model we just trained, we need to export it.

current_directory = os.getcwd()
print(current_directory)

# +
output_folder = current_directory+'/outputs/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

learn.export(output_folder + 'image_classif_model_f44.pkl')
# -

# ### Metrics visualization

# #### 1. On the Azure portal

# To visualize the training and validation metrics on the portal, we need to "attach" them to the current experiment's run

# +
training_losses = [x.numpy().ravel()[0] for x in learn.recorder.losses]
accuracy_list = [x[0].numpy().ravel()[0] for x in learn.recorder.metrics]

runner.log_list('training_loss', training_losses)
runner.log_list('validation_loss', learn.recorder.val_losses)
runner.log_list('accuracy', accuracy_list)
runner.log_list('learning_rate', learn.recorder.lrs)
# -

# These metrics are now part of the experiment (through `runner.log_list`). We can then visualize them on the Azure portal, by clicking on the latest run number or by running the `runner.get_portal_run()` command below.
#
# <img src="screenshots/experiment_run_recorded_metrics.jpg" width="800">
#
# Note that the "learning_rate" plot indeed shows the variation of this metric over one cycle only.

print(runner.get_portal_url())

# `runner.log_list` gives access to the training and validation metrics on the Azure portal. We can also look at them on our local machine using not only fast.ai plotting functions, but also Tensorboard.

# #### 2. On our local machine, using fast.ai

# +
# learn.recorder.plot() # -- Need a bigger dataset for that
# -

# #### 3. On our local machine, using Tensorboard
#
# [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) is a suite of visualization tools developed by Google as part of their TensorFlow framework. It provides an interactive interface and allows users to compare runs in an esay fashion.
#
# While we trained our model from a fast.ai's / pytorch's pre-trained model, it is still possible for us to visualize the training and validation metrics with Tensorboard.
#
# For that, the content of `runner` must be made available to Tensorboard. Thus, we first need to create a folder that will contain the run history. From there, we will be able to export that content to that folder.

# Export Run History to Tensorboard logs
logdir = 'exportedTBlogs'
log_path = os.path.join(os.getcwd(), logdir)
try:
    os.stat(log_path)
except os.error:
    os.mkdir(log_path)
print("Tensorboard logs will be stored in {}".format(log_path))

# +
# # ! pip freeze | grep "tensorboard"
# -

# Export run history for the project
export_to_tensorboard(runner, logdir)

# Now that the run history is available, we can start Tensorboard, and access the metrics in a web browser.

# The Tensorboard constructor takes an array of runs
tb = Tensorboard([], local_root=logdir, port=6006)

# If successful, start() returns a string with the URI of the instance
uri = tb.start()

# Tensorboard can be opened:
#  - Manually, by clicking on the URI printed above
#  - Directly, using python

webbrowser.open(uri)

#  - Or by running the following command in a terminal window:
# `tensorboard --logdir=<content_of_log_path>`
#
# This should print, on the terminal, the same URI as `tb.start()`.
#
# <img src="screenshots/tensorboard.jpg" width="500">

# Tensorboard is especially convenient to compare several runs in the same graph.

# ### Run visualization
# Now, that we have trained the model, we can close the run using `runner.complete()`. Results can then be captured and finalized. `runner` returns a URL that allows us to see them on the Azure portal.

runner.complete()

runner

# ### Model deployment on Azure

# #### 1. Model registration
#
# Our final goal is to deploy the model we trained as a web service. To do so, we need to first register it in our workspace, i.e. place it in our workspace's model registry. We do this by using `runner.upload_file()` and `runner.register_model()`, which takes:
# - a model_name that represents well what our model does
# - and the model_path of the model relative to the run.
#
# Other methods exist to load the model and register it all at once, but using `runner.upload_file()` and specifying the `outputs/` folder allow us to check the presence of the uploaded model on the Azure portal.

runner.upload_file(name='outputs/image_classif_model_f44.pkl', path_or_stream=current_directory+'/outputs/image_classif_model_f44.pkl')

# Register the model with the workspace
model = runner.register_model(model_name='image_classif_model_f44', model_path='outputs/image_classif_model_f44.pkl')
# !!! We need to make sure that the model name we use here is the same as in the scoring script below !!!

# Now that the model is uploaded and registered, we can see it on the Azure platform, under `Outputs` and `Models`
#
# <img src="screenshots/uploaded_model.jpg" width="800">
# <img src="screenshots/models.jpg" width="800">

# We can also check that it is programatically accessible

print("Model:\n --> Name: {}\n --> ID: {}\n --> Path:{}"
      .format(model.name, model.id, model._get_model_path_remote(model.name, model.version, ws)))

runner.get_file_names()

# If we are also interested in verifying which model we uploaded, we can download it to our local machine

model.download()

# #### 2. Scoring script
# For the web service to return predictions on a given input image, we need to provide it with instructions on how to use the model we just registered. These instructions are stored in the scoring script.
#
# This script must contain two required functions, `init()` and `run(input_data)`:
# - In the `init()` function, we typically load the model into a global object. This function is executed only once when the Docker container is started.
# - In the `run(input_data)` function, the model is used to predict a value based on the input data. The input and output to `run` typically use JSON as serialization and de-serialization format but we are not limited to that.
#
# This file must also be stored in the current directory.

scoring_script = "score.py"

# + {"magic_args": "$scoring_script", "language": "writefile"}
# # Copyright (c) Microsoft. All rights reserved.
# # Licensed under the MIT license.
#
# import json
# import numpy as np
# from azureml.core.model import Model
# from fastai.vision import *
# from fastai.vision import Image as FImage
#
# def init():
#     global model
#     model_path = Model.get_model_path(model_name='image_classif_model_f44')
#     actual_path, actual_file = os.path.split(model_path)
#     model = load_learner(path=actual_path, fname=actual_file)
#
#
# def run(raw_data):
#     
#     result = []
#
#     all_data = json.loads(raw_data)['data']
#     
#     try:
#         highest_dimension = len(all_data[0][0][0])
#         # Case of several images -- batch processing
#     except Exception as e:
#         all_data = [all_data]
#         # Case of a single image -- process it as a batch of 1 image
#     
#     for data in all_data:
#         try:
#             data_arr = np.asarray(data)
#             data_tensor = torch.Tensor(data_arr)
#             data_image = FImage(data_tensor)
#             pred_class, pred_idx, outputs = model.predict(data_image)
#             result_dict = {"label": str(pred_class), "probability": str(float(outputs[pred_idx]))}
#             result.append(result_dict)
#         except Exception as e:
#             result_dict = {"label": str(e), "probability": ''}
#             result.append(result_dict)
#                 
#     return result
#
# -

# #### 3. Environment setup

# In order to make predictions on the Azure platform, it is critical to create an environment similar to the one we used to train the model. For this, we use a [Docker container](https://www.docker.com/resources/what-container). We configure it via a yaml file that will contain all the conda dependencies we relied on on our local machine.

# +
# Create an empty conda environment and add the scikit-learn package
conda_filename = "myenv.yml"  # this file can only be save in the current directory

conda_env = CondaDependencies()
conda_env.add_conda_package("scikit-learn")
conda_env.add_channel("pytorch")
conda_env.add_channel("conda-forge")
conda_env.add_channel("fastai")
conda_env.add_conda_package("pytorch-cpu")
conda_env.add_conda_package("torchvision")
conda_env.add_conda_package("spacy=2.0.18")
conda_env.add_conda_package("fastai=1.0.44")
conda_env.add_conda_package("dataclasses")

# Display the environment
print(conda_env.serialize_to_string())

# Write the environment to disk
conda_env.save_to_file(base_directory='./', conda_file_path=conda_filename)
# -

# Create a configuration object indicating how our deployment container needs to be created
image_config = ContainerImage.image_configuration(execution_script=scoring_script,
                                                  runtime="python", 
                                                  conda_file="myenv.yml")

# #### 4. Compute target (i.e. computational resources)

# In addition to the container, we also need to describe the type of compute we want to allocate for our webservice. In this example, we are using an [Azure Container Instance](https://azure.microsoft.com/en-us/services/container-instances/) which is good for quick and cost-effective development/test deployment scenarios.
#
# To set ACI instances up properly, we also need to indicate the number of cores we want to run and the amount of memory we need. Optional tags and descriptions are also available for us to identify the instances in AzureML when viewing the Compute tab in the Azure Portal.
#
# For production workloads, it is better to use [Azure Kubernentes Service (AKS)](https://azure.microsoft.com/en-us/services/kubernetes-service/) instead. In this notebook *to be created* >> ADD LINK HERE we demonstrate how to proceed.

# Create a deployment configuration with 1 CPU and 5 gigabyte of RAM
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, 
                                                memory_gb=5, 
                                                tags={'webservice': 'image classification model (fastai1.0.44)'}, 
                                                description='This service classifies images into 4 different groups (bowl, cup, cutlery and plate).')

# #### 5. Web service deployment

# The final step to deploying our webservice is to call `WebService.deploy_from_model()`. This function uses the deployment and image configurations we created above to perform the following:
#
# - Build a docker image
# - Deploy to the docker image to an Azure Container Instance
# - Copy our model files to the Azure Container Instance
# - Call the `init()` function in our scoring file
# - Provide an HTTP endpoint for scoring calls
#
# The `deploy_from_model` method requires the following parameters:
#
# - workspace: the workspace containing the service
# - name: a unique named used to identify the service in the workspace
# - models: an array of models to be deployed into the container
# - image_config: a configuration object describing the image environment
# - deployment_config: a configuration object describing the compute type
#
# Note: The web service creation can take **several minutes**.

# Creation of the Docker container image and deployment of the web service
service_name = 'image-classif-websvc-f44'
service = Webservice.deploy_from_model(name=service_name,
                                       workspace=ws,
                                       image_config=image_config,
                                       deployment_config=aci_config,
                                       models=[model])

service.wait_for_deployment(show_output=True)

# When successful, we expect to see the following:
#
# `Creating image
# Image creation operation finished for image <service_name>, operation "Succeeded"
# Creating service
# Running .....
# SucceededACI service creation operation finished, operation "Succeeded"`
#
# In the case where the deployment is not successful, we can look at the image and service logs to debug.

# Access to the build logs
print(ws.images[service_name].image_build_log_uri)

# +
# Access to the service logs
# print(service.get_logs())
# -

print("Service {} is _{}_ and available at {}".format(service.name, service.state, service.scoring_uri))

# #### 6. Testing of the service

# To make sure that our service is working as expected, let's test it. We first need to retrieve a test image and to pre-process it into the format expected by our model.

# ##### 6a. Using the `run` API

# A service typically expects input data to be in a JSON serializable format, which fast.ai's image objects are not. Here, we use our own `image2json()` function to transform these images into lists of floats and then into strings.

# Conversion of image into JSON serializable object
images_fname_list = ['test_images/msft-plastic-bowl20170725152154282.jpg', 'test_images/msft-paper-plate20170725132850611.jpg']
test_samples, all_images = image2json(images_fname_list, current_directory)

# Prediction using the deployed model
result = service.run(test_samples)  # This takes several seconds

# Plots of the results
actual_labels = ['plastic-bowl', 'paper-plate']
for k in range(0, len(result)):
    title = "{}/{} - {}%".format(actual_labels[k], result[k]['label'], 
                                 round(100.*float(result[k]['probability']), 2))
    all_images[k].show(title=title)

# ##### 6b. Via a raw HTTP request

# +
# Send the same test data
headers = {'Content-Type':'application/json'}

resp = requests.post(service.scoring_uri, test_samples, headers=headers)

print("POST to url", service.scoring_uri)
print("prediction:", resp.text)
# -

# ##### 6c. Using a user interface

# As we discussed above, Azure Container Instances are typically used to develop and test deployments. They are however not suited for production, or calls to the service outside of the run context. For this to be possible, we need to deploy to Azure Kubernetes Service. An example of this is available in this notebook, where we show how to create a web app (???) with a user interface. >>> CREATE IT AN LINK TO IT HERE

# ### Clean up

# #### Service termination
#
# In a real-life scenario, it is likely that the service we created would need to be up and running at all times. However, in the present demonstrative case, and now that we have verified that our service works, we can delete it.

service.delete()

# #### Tensorboard closure

# We were able to track the evolution of the training and validation loss and accuracy during training. Now that our model is deployed, we can close our Tensorboard session.

tb.stop()
