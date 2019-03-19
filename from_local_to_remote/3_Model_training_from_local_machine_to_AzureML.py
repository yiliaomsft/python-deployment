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

# # Model training locally and in the cloud with the same code
#
# In this notebook, we will train a simple model, first on our local machine, and then on AzureML, with the exact same code.
#
# >> ADD BLURB ON THE DIFFERENT STEPS

# ### Library import

# +
# For automatic reloading of modified libraries
# %reload_ext autoreload
# %autoreload 2

# Regular python libraries
from datetime import datetime
import json
import os
import papermill as pm
from sys import platform
from utils import *

from string import digits

# fast.ai
from fastai.vision import *
import torchvision.models as models

# Azure
import azureml.core
from azureml.core.conda_dependencies import CondaDependencies 
# -

# ### Data retrieval

# In this notebook, we will use the pets dataset provided by fast.ai.
# All the images are stored in a single `images/` folder, and each image filename contains the image's label. We can then use fast.ai's `ImageDataBunch.from_name_re()` method to extract the images content and corresponding labels using a regular expression.
# >> SEE IF WE WANT TO REPLACE IT BY SOMETHING ELSE

path = untar_data(URLs.PETS)
path_anno = path/'annotations'
path_img = path/'images'

print("> Data are now stored in {}".format(path))
print("> They are composed of: {}".format(path.ls()))

# A few filename examples
fnames = get_image_files(path_img)
fnames[:5]

# Images are in stored in alphabetical order. For proper training, it is important to shuffle them, and to use a random seed for it.

# +
pattern = r'/([^/]+)_\d+.jpg$'

# The filenames are of the form:
# - starts with a forward slash,
# - is followed by many characters that are not forward slashes
# - then followed by an underscore and one or more digits
# - has a ".jpg" extension

# +
# By default, fast.ai uses batch sizes of 64 images
# Here, we are using batches of 16 to ensure that we have enough memory to process the data

# We load the data and parse the labels from the image filenames
# We transform the images such that they now have a 224x224 shape

new_batch_size = 16
data = ImageDataBunch.from_name_re(path=path_img, fnames=fnames, pat=pattern, 
                                   ds_tfms=get_transforms(), size=224, 
                                   bs=new_batch_size).normalize(imagenet_stats)
# -

data.train_ds

print(data.classes)
print(len(data.classes))

# This dataset contains 5912 images, grouped into 37 classes. These images have now been transformed to have a 224x224 shape. The figure below shows a few examples of them.

data.show_batch(rows=3, figsize=(7,6))

# ### Model training - Locally
#
# Here, we use resnet34, i.e. a [residual neural network](https://arxiv.org/pdf/1512.03385.pdf) that contains 34 hidden layers, and leverage the [default settings](https://pytorch.org/docs/stable/torchvision/models.html#id3) implemented by Pytorch.

# +
mdl_object = models.resnet34 # any model among alexnet, resnet18/34/50/101/152, squeezenet1.0/1.1, densenet121/161/169/201, vgg16/19

if platform == 'win32':
    learn = custom_create_cnn(data, mdl_object, metrics=error_rate)
else:
    learn = create_cnn(data, mdl_object, metrics=error_rate)
# -

print(datetime.now())
learn.fit_one_cycle(4)  # This typically takes ~ 2h15min on a CPU

# Training this model on a single CPU machine typically takes about 2h15min.

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

interp.most_confused(min_val=2)

# +
output_folder = os.getcwd() + '/outputs/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

learn.export(output_folder + 'image_classif_model_cats_dogs.pkl')

# +
#Record of the model metrics, so they are accessible when this notebook gets run remotely
training_losses = [x.numpy().ravel()[0] for x in learn.recorder.losses]
accuracy_list = [x[0].numpy().ravel()[0] for x in learn.recorder.metrics]

pm.record('training_loss', training_losses)
pm.record('validation_loss', learn.recorder.val_losses)
pm.record('accuracy', accuracy_list)
pm.record('learning_rate', learn.recorder.lrs)

# +
# learn.recorder.plot()
# -

# Now that we have trained our model on our local machine, let's reuse this same notebook and launch its execution on AzureML. For this, we will use the Launch_notebook_on_AzureML.ipynb notebook. << ADD LINK HERE >>

# +
#Reference notebook: https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb
