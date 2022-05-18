#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")

import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import os, time, uuid

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials


# In[2]:


TRAINING_ENDPOINT = "https://dungdocustomvision.cognitiveservices.azure.com/"
training_key = "41877457e30e426a99e1c305205bfc22"
training_resource_id = '/subscriptions/21c53bc7-9f96-4753-9901-99cd641ad4e7/resourceGroups/ODL-AIND-195702/providers/Microsoft.CognitiveServices/accounts/dungdocustomvision'


# In[3]:


# Replace with valid values
PREDICTION_ENDPOINT = "https://dungdocustomvision-prediction.cognitiveservices.azure.com/"
prediction_key = "bfd558bd968a4ee8ad3f7f09cb6cfd70"
prediction_resource_id = "/subscriptions/21c53bc7-9f96-4753-9901-99cd641ad4e7/resourceGroups/ODL-AIND-195702/providers/Microsoft.CognitiveServices/accounts/dungdocustomvision-Prediction"


# In[4]:


training_credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, training_credentials)


# In[5]:


trainer.api_version


# In[6]:


# Create a new project
print ("Training project created. Proceed to the next cell.")
project_name = uuid.uuid4()
project = trainer.create_project(project_name)


# In[7]:


project.as_dict()


# In[8]:


lighter_check = trainer.create_tag(project.id, "LighterCheck")


# In[9]:


luggage_check = trainer.create_tag(project.id, "LuggageCheck")


# In[10]:


local_image_path = '/home/workspace/ImageToTrain'


# In[11]:


# Some code is taken from Azure SDK Sample
def upload_images_for_training(local_project_id, local_img_folder_name, image_tag_id):
    image_list = []
    files = os.listdir(os.path.join (local_image_path, local_img_folder_name))
    for file in files:
        full_path = os.path.join(local_image_path, local_img_folder_name, file)
        if os.path.isfile(full_path) and full_path.endswith('.jpg'):
            with open(os.path.join (local_image_path, local_img_folder_name, file), "rb") as image_contents:
                image_list.append(ImageFileCreateEntry(name=file, contents=image_contents.read(), tag_ids=[image_tag_id]))
                
    upload_result = trainer.create_images_from_files(local_project_id, ImageFileCreateBatch(images=image_list))
    if not upload_result.is_batch_successful:
        print("Image batch upload failed.")
        for image in upload_result.images:
            print("Image status: ", image.status)
        exit(-1)
    return upload_result


# In[12]:


lighter_upload_result = upload_images_for_training(project.id, 'lighter_folder', lighter_check.id)


# In[13]:


lighter_upload_result.is_batch_successful


# In[14]:


luggage_check_result = upload_images_for_training(project.id, 'luggage_folder', luggage_check.id)


# In[15]:


luggage_check_result.is_batch_successful


# In[16]:


iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    print ("Waiting 10 seconds...")
    time.sleep(10)


# In[17]:


iteration.as_dict()

iteration_list = trainer.get_iterations(project.id)
for iteration_item in iteration_list:
    print(iteration_item)
    
model_perf = trainer.get_iteration_performance(project.id, iteration_list[0].id)

model_perf.as_dict()


# In[18]:


# Setting the Iteration Name, this will be used when Model training is completed
# Please choose a name favorable to you.
publish_iteration_name = "lighter-check"


# In[19]:


# The iteration is now trained. Publish it to the project endpoint
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print ("Done!")


# In[ ]:




