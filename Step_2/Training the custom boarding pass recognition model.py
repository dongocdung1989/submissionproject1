#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")

import os
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.formrecognizer import FormRecognizerClient
from azure.ai.formrecognizer import FormTrainingClient
from azure.core.credentials import AzureKeyCredential


# In[2]:


AZURE_FORM_RECOGNIZER_ENDPOINT = "https://formrecognizerdungdo.cognitiveservices.azure.com/"
AZURE_FORM_RECOGNIZER_KEY = "11fa6a148a0d4c48845d7ee84c5479bd"


# In[3]:


endpoint = AZURE_FORM_RECOGNIZER_ENDPOINT
key = AZURE_FORM_RECOGNIZER_KEY

form_training_client = FormTrainingClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# In[4]:


saved_model_list = form_training_client.list_custom_models()


# In[5]:


trainingDataUrl = "https://dungdoazure.blob.core.windows.net/boardingpasscontainer?si=all&sv=2020-08-04&sr=c&sig=bNdl9Iet9P8f2jDd%2BMbKJlEpto9h2VGlpgtfn0NQK9Q%3D"


# In[6]:


training_process = form_training_client.begin_training(trainingDataUrl, use_training_labels=False)
custom_model = training_process.result()


# In[7]:


custom_model


# In[8]:


custom_model.model_id


# In[9]:


custom_model.status


# In[10]:


custom_model.training_started_on


# In[11]:


custom_model.training_completed_on


# In[12]:


custom_model.training_documents


# In[13]:


for doc in custom_model.training_documents:
    print("Document name: {}".format(doc.name))
    print("Document status: {}".format(doc.status))
    print("Document page count: {}".format(doc.page_count))
    print("Document errors: {}".format(doc.errors))


# In[14]:


custom_model.properties


# In[15]:


custom_model.submodels


# In[16]:


for submodel in custom_model.submodels:
    print(
        "The submodel with form type '{}' has recognized the following fields: {}".format(
            submodel.form_type,
            ", ".join(
                [
                    field.label if field.label else name
                    for name, field in submodel.fields.items()
                ]
            ),
        )
    )


# In[17]:


custom_model.model_id


# In[18]:


custom_model_info = form_training_client.get_custom_model(model_id=custom_model.model_id)
print("Model ID: {}".format(custom_model_info.model_id))
print("Status: {}".format(custom_model_info.status))
print("Training started on: {}".format(custom_model_info.training_started_on))
print("Training completed on: {}".format(custom_model_info.training_completed_on))


# In[19]:


form_recognizer_client = FormRecognizerClient(endpoint=endpoint, credential=AzureKeyCredential(key))
new_test_url = "https://dungdoazure.blob.core.windows.net/boardingpasscontainer/boarding_pass_Dung%20-%20test%20train.pdf?si=all&sv=2020-08-04&sr=b&sig=2V8wLjYa6cZyzIz2JcGcxCfOPOWGhXhpVAVDNa4zsV0%3D"

labeled_training_process = form_training_client.begin_training(trainingDataUrl, use_training_labels=True)
labeled_custom_model = labeled_training_process.result()
labeled_custom_test_action = form_recognizer_client.begin_recognize_custom_forms_from_url(model_id=labeled_custom_model.model_id, form_url=new_test_url)


# In[21]:


labeled_custom_test_action.status()


# In[22]:


labeled_custom_test_action_result = labeled_custom_test_action.result()


# In[23]:


for recognized_content in labeled_custom_test_action_result:
    print("Form type: {}".format(recognized_content.form_type))
    for name, field in recognized_content.fields.items():
        print("Field '{}' has label '{}' with value '{}' and a confidence score of {}".format(
            name,
            field.label_data.text if field.label_data else name,
            field.value,
            field.confidence
        ))


# In[ ]:





# In[ ]:




