#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import FormRecognizerClient

AZURE_FORM_RECOGNIZER_ENDPOINT = "https://dungdoformrecognizer.cognitiveservices.azure.com/"
AZURE_FORM_RECOGNIZER_KEY = "8adca8f86cc44292864b1796b8960d77"

endpoint = AZURE_FORM_RECOGNIZER_ENDPOINT
key = AZURE_FORM_RECOGNIZER_KEY

form_recognizer_client = FormRecognizerClient(endpoint=endpoint, credential=AzureKeyCredential(key))

content_url = "https://raw.githubusercontent.com/dongocdung1989/Project1AutoKios/main/digital_id_DungDo.PNG"
id_content_from_url = form_recognizer_client.begin_recognize_identity_documents_from_url(content_url)

collected_id_cards = id_content_from_url.result()
collected_id_cards

type(collected_id_cards[0])


# In[2]:


def get_id_card_details(identity_card):
    first_name = identity_card.fields.get("FirstName")
    if first_name:
        print("First Name: {} has confidence: {}".format(first_name.value, first_name.confidence))
    last_name = identity_card.fields.get("LastName")
    if last_name:
        print("Last Name: {} has confidence: {}".format(last_name.value, last_name.confidence))
    document_number = identity_card.fields.get("DocumentNumber")
    if document_number:
        print("Document Number: {} has confidence: {}".format(document_number.value, document_number.confidence))
    dob = identity_card.fields.get("DateOfBirth")
    if dob:
        print("Date of Birth: {} has confidence: {}".format(dob.value, dob.confidence))
    doe = identity_card.fields.get("DateOfExpiration")
    if doe:
        print("Date of Expiration: {} has confidence: {}".format(doe.value, doe.confidence))
    sex = identity_card.fields.get("Sex")
    if sex:
        print("Sex: {} has confidence: {}".format(sex.value, sex.confidence))
    address = identity_card.fields.get("Address")
    if address:
        print("Address: {} has confidence: {}".format(address.value, address.confidence))
    country_region = identity_card.fields.get("CountryRegion")
    if country_region:
        print("Country/Region: {} has confidence: {}".format(country_region.value, country_region.confidence))
    region = identity_card.fields.get("Region")
    if region:
        print("Region: {} has confidence: {}".format(region.value, region.confidence))


# In[3]:


for index_id, id_card in enumerate(collected_id_cards):
    print("Displaying identity card details ....... # {}".format(index_id+1))
    get_id_card_details(id_card)
    print("---------------- EOL -------------------------")


# In[ ]:




