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

# ## Deployment of a model as a service with Azure Kubernetes Service
#
#
# At the end of this notebook, we will have learned how to:
#
# - Deploy an already trained model as a web service using [Azure Kubernetes Service](https://docs.microsoft.com/en-us/azure/aks/concepts-clusters-workloads)
# - Monitor our new service
# - Set up a user interface which calls our service
#
# We will use the model we trained locally and registered on the Azure platform, in the prior notebook (ADD LINK HERE).

# ### Pre-requisites
#
# It is assumed here that our local environment and Azure subscription are already set up (cf. [details](LINK TO PRIOR notebook)).

# ### Library import

# +
# For automatic reloading of modified libraries
# %reload_ext autoreload
# %autoreload 2

# Regular python libraries
import json
import numpy as np
import os
import requests
from string import digits
from sys import platform
import webbrowser
from utils import *

# fast.ai
from fastai.vision import *
import torchvision.models as models

# Azure
import azureml.core
from azureml.core import Workspace
from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.image import ContainerImage
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import AksWebservice, Webservice
# -

# ### Azure workspace

# Let's start by loading the working space which information we saved in our aml_config/config.json file

ws = Workspace.from_config()

# Let's check that the workspace is properly loaded

# Print the workspace attributes
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

# ### Model retrieval
#
# To deploy a model as a service, we need to be aware of what that model is. In the prior notebook, we defined our `model` object as the output of the `runner.register_model()` method.
#
# Here, since we have already registered our model, we only need to get it from our workspace context. Note: if several versions of a same model exist, the version that gets retrieved is the latest one.

# List of all models registered under this workspace
ws.models

# Model we are interested in
model = ws.models['image_classif_model_f44']

print("Existing model:\n --> Name: {}\n --> Version: {}\n --> ID: {} \
      \n --> Creation time: {}\n --> URL: {}\n --> Properties: {}"
      .format(model.name, model.version, model.id, model.created_time, 
              model.url, model.properties))

# ### Model deployment
#
# As for the deployment on Azure Container Instances, we will use Docker containers. For this, we need a yaml file listing the libraries to import, and a scoring script. These are exactly the same as the ones we generated in the previous notebook, but just to remind ourselves:

# #### Environment setup

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

# Write the environment to disk
conda_env.save_to_file(base_directory='./', conda_file_path=conda_filename)

# +
^^^ REPLACE THE ABOVE WITH

conda_env = CondaDependencies(conda_dependencies_file_path='../../environment.yaml')  -- fix the path
conda_env.add_conda_package("fastai=1.0.44") 
conda_env.add_conda_package("dataclasses")


# Help from: https://github.com/Microsoft/Recommenders/blob/danielsc/azureml/notebooks/02_train_in_the_cloud/train_fastai_on_aml.ipynb
^^ SEE IF THAT SUFFICES
# -

# #### Scoring script

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
# -

# ### Docker image
#
# We proceed slightly differently here in the creation of the image. This way gives us access to the Docker image object. Thus, if the service deployment fails, but the Docker image gets deployed successfully, we can try deploying the service again, without having to create a new image.

image_config = ContainerImage.image_configuration(execution_script = "score.py",
                                                  runtime = "python",
                                                  conda_file = "myenv.yml",
                                                  description = "Image with fast.ai Resnet18 model (fastai 1.0.44)",
                                                  tags = {'area': "tableware", 'type': "CNN resnet"}
                                                 )

image = ContainerImage.create(name = "image-classif-resnet18-f44-aks",
                              # this is the model object
                              models = [model],
                              image_config = image_config,
                              workspace = ws)

# %%time
image.wait_for_creation(show_output = True)

print(ws.images["image-classif-resnet18-f44-aks"].image_build_log_uri)

# +
# ws.compute_targets['cpucluster'].type
# -

# ### Azure Kubernetes Service
#
# For our service to be able to score new images, it needs not only a model, but also computational resources.
#
# If we already have a Kubernetes-managed cluster in this workspace, we can find and use it, otherwise, we can:
# - either create a new one
# - attach a cluster that we have under our subscription and resource group, but not under this workspace.
#
# Note: The name we give to our compute target must be between 2 and 16 characters long.

# #### 1. AKS compute target creation

# +
virtual_machine_type = 'gpu'
aks_name = 'imgclass-aks-{}'.format(virtual_machine_type)

if aks_name not in ws.compute_targets:
    compute_name = os.environ.get("AKS_COMPUTE_CLUSTER_NAME", aks_name)

    if virtual_machine_type == 'gpu':
        vm_size_name ="Standard_NC6"  #<<<< SEE WHAT TO PUT HERE
    else:
        vm_size_name = "Standard_D3_v2"
    vm_size = os.environ.get("AKS_COMPUTE_CLUSTER_SKU", vm_size_name)   # ISSUES HERE -- IMAGES DON'T GET CRAETED ANYMORE

print("Our AKS computer target's name is: {}".format(aks_name))

# +
# aks_name = 'img-classif-aks'

# Use the default configuration (can also provide parameters to customize)
if aks_name not in ws.compute_targets:
    prov_config = AksCompute.provisioning_configuration(vm_size = vm_size)
# -

# %%time
if aks_name not in ws.compute_targets:
    # Create the cluster
    aks_target = ComputeTarget.create(workspace = ws, 
                                      name = aks_name, 
                                      provisioning_configuration = prov_config)
    aks_target.wait_for_completion(show_output = True)
else:
    # Retrieve the already existing cluster
    aks_target = ws.compute_targets[aks_name]
    print("We retrieved the {} AKS compute target".format(aks_target.name))

# This new compute target can be seen on the Azure portal, under the Compute tab.
#
# <img src="screenshots/aks_compute_target.jpg" width="900">

# Check provisioning status
print("The AKS compute target provisioning {} -- There were {} errors"
      .format(aks_target.provisioning_state.lower(), aks_target.provisioning_errors))

# #### 2. Monitoring activation
#
# Once our webapp is up and running, it will be very important to monitor it, and measure the amount of traffic it gets, how long it takes to respond, the type of exceptions that get raised, etc. We will do so through [Application Insights](https://docs.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview), which is an application performance management service. To enable it, we first need to update our AKS configuration file:

#Set the web service configuration
aks_config = AksWebservice.deploy_configuration(enable_app_insights=True)

# #### 3. Model deployment

if aks_target.provisioning_state== "Succeeded": 
    aks_service_name ='aks-image-classif-web-svc'
    aks_service = Webservice.deploy_from_image(workspace = ws, 
                                               name = aks_service_name,
                                               image = image,
                                               deployment_config = aks_config,
                                               deployment_target = aks_target
                                               )
    aks_service.wait_for_deployment(show_output = True)
    print(aks_service.state)
else:
    raise ValueError("AKS provisioning failed.")

>>> CHECK IF ISSUE IS BECAUSE OF FASTAI VERSION 

# +
# print(aks_service.get_logs())
# -

# The new deployment can be seen on the portal, under the Deployments tab.
#
# <img src="screenshots/aks_webservice.jpg" width="900">

# #### 4. Testing of the service

# ##### 4a. Using the `run` API
#
# As in the previous notebook, we first convert our test images into JSON serializable data

# Conversion of image into JSON serializable object
current_directory = os.getcwd()
images_fname_list = ['test_images/msft-plastic-bowl20170725152154282.jpg', 'test_images/msft-paper-plate20170725132850611.jpg']
test_samples, all_images = image2json(images_fname_list, current_directory)

# Prediction using the deployed model
if aks_service.state == "Healthy":
    result = aks_service.run(input_data=test_samples)  # This takes several seconds
else:
    raise ValueError("Service deployment isn't healthy, can't call the service")

# Plots of the results
if aks_service.state == "Healthy":
    actual_labels = ['plastic-bowl', 'paper-plate']
    for k in range(0, len(result)):
        title = "{}/{} - {}%".format(actual_labels[k], result[k]['label'], 
                                     round(100.*float(result[k]['probability']), 2))
        all_images[k].show(title=title)

# ##### 4b. Via an HTTP request
#
# Deploying a model as a web service creates a REST API. We can then call that API outside of our current workspace and notebook. For that, we need:
# - The web service's URL
# - One of the authentication keys.
#
# By default, authentication is enabled when deploying models to Azure Kubernetes Service. We can the retrieve both pieces of information easily. 

# +
# Service URL
service_url = aks_service.scoring_uri
print("Send POST requests to: {}".format(service_url))

# Authentication keys
primary, secondary = aks_service.get_keys()
print("Keys to use when calling the service from an external app: {}".format([primary, secondary]))
# -

# Send the same test data
if aks_service.state == "Healthy":

    key = aks_service.get_keys()[0]
    # Set the content type
    headers = { 'Content-Type':'application/json' }
    # Set the authorization header
    headers['Authorization']=f'Bearer {key}'
    
    # Send the request
    resp = requests.post(service_url, test_samples, headers=headers)

    print("Predictions: {}".format(resp.text))

# ##### 4c. Using a user interface -- Locally
#
# This notebook is accompanied by 3 files:
# - file_uploader.py
# - templates/index.html
# - templates/template.html
#
# They construct a Flask application that will allow us to test that our service is working as we expect.
#
# We can run this test in 2 different ways:
# 1. From a terminal window, in our conda environment
# 2. From within this notebook
#
# ###### - From the terminal -
# To run the Flask application from our local machine, we need to:
# - Copy the 3 files on our machine
# - Run `python file_uploader.py 'webservice_url' 'authentication_key'`, where <font color=green>webservice_url</font> and <font color=green>authentication_key</font> should be replaced by the values we obtained above
#
# This returns a URL (typically http :// 127.0.0.1:5000). Clicking on it brings us to a file uploader webpage.
#
# If our service works as expected, after a few seconds, we can see the results presented in a table. CHANGE THE RESULTS WHEN HAVE THOSE WITH NEW DATASET
#
# <img src="screenshots/file_uploader_webpage.jpg" width="500" align="left">
# <img src="screenshots/predictions.jpg" width="300" align="center">
#
# Notes:
# - Depending on the size of the uploaded images, the service may or may not provide a response. It is best to send a few small images, i.e. 3-5 images a few kB each.
# - The uploader function creates an uploads/ folder in our working directory, which contains the images we uploaded.
#
# ###### - From this notebook -
# Here, we use a built-in magic command `%run`. The URL it returns is the same as above, so we can enter it directly in a browser. The experience is then the same.
#
# To end the test, we just need to hit "Ctrl+C" in the terminal or the "Stop" (square) button in the notebook.

# Built-in magic command to run our Flask application
# Note: "$" in front of our variables names returns their content
# %run -i file_uploader.py $service_url $key

# +
# Let's not forget to end the test (Ctrl+C or Stop button)
# -

# ##### 4d. Using a user interface - Online

>>> ADD SOMETHING ABOUT APP LOGICS HERE

aks_service.swagger_uri



# #### 5. Service telemetry in [Application Insights](https://docs.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview)
#
# In the [Azure portal](https://portal.azure.com):
# - Let's navigate to "All resources"
# - Select our subscription and resource group that contain our workspace
# - Select the Application Insights type associated with our workspace
#   * _If we have several, we can still go back to our workspace (in the portal) and click on "Overview" - This shows the elements associated with our workspace, in particular our Application Insights_
# - Click on the App Insights resource: There, we can see a high level dashboard with information on successful and failed requests, server response time and availability
# - Select "Analytics/Request count" in the "View in Analytics " drop-down: This displays the specific query ran against the service logs to extract the number of executed requests (successful or not).
# - Still in the "Logs" page, click on the eye icon next to "requests" on the "Schema"/left pane, and on "Table" on the right one:
#   * This shows the list of calls to the service, with their success statuses, durations, and other metrics. This table is especially useful to investigate problematic requests.
#   * Results can also be visualized as a graph by clicking on the "Chart" tab. Metrics are plotted by default, but we can change them by clicking on one of the field name drop-downs.
# - Navigate across the different queries we ran through the different "New Query X" tabs.
#
# <img src="screenshots/webservice_performance_metrics.jpg" width="400" align="left">
# <img src="screenshots/application_insights_all_charts.jpg" width="500" align="center">
#
# <br><br>
# <img src="screenshots/failures_requests_line_chart.jpg" width="500" align="right">
# <img src="screenshots/all_requests_line_chart.jpg" width="400" align="left">
#
# <br><br>
# <img src="screenshots/logs_failed_request_details.jpg" width="500" align="right">
# <img src="screenshots/success_status_bar_chart.jpg" width="450" align="left">
#
#

# ### Clean up

# In a real-life scenario, it is likely that the service we created would need to be up and running at all times. However, in the present demonstrative case, and now that we have verified that our service works, we can delete it as well as all the resources we used.

# #### Application Insights deactivation

aks_service.update(enable_app_insights=False)

# #### Service termination

aks_service.delete()

# #### Image deletion

image.delete()
