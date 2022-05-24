#!/usr/bin/env python
# coding: utf-8

# ### Please install the required Python modules/SDKs

# In[8]:


get_ipython().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")
get_ipython().system('pip install Pillow==8.4')
import io
import datetime
import pandas as pd
from PIL import Image
import requests
import io
import glob, os, sys, time, uuid

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw

from video_indexer import VideoIndexer
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import TrainingStatusType
from msrest.authentication import CognitiveServicesCredentials


# In[9]:


CONFIG = {
    'SUBSCRIPTION_KEY': '2e22306c33174c65821041bebacf5042',
    'LOCATION': 'trial',
    'ACCOUNT_ID': '3d321732-f1e5-4517-a86d-515dfa1109f0'
}

video_analysis = VideoIndexer(
    vi_subscription_key=CONFIG['SUBSCRIPTION_KEY'],
    vi_location=CONFIG['LOCATION'],
    vi_account_id=CONFIG['ACCOUNT_ID']
)
video_analysis.check_access_token()


# In[10]:


video_id = '7fc6b7ff52'
video_analysis.get_video_info(video_id)
info = video_analysis.get_video_info(video_id, video_language='English')
if len(info['videos'][0]['insights']['faces'][0]['thumbnails']):
    print("We found {} faces in this video.".format(str(len(info['videos'][0]['insights']['faces'][0]['thumbnails']))))


# In[11]:


info['videos'][0]['insights']['faces'][0]['thumbnails']


# In[12]:


if len(info['videos'][0]['insights']['faces'][0]['thumbnails']):
    print("We found {} faces in this video.".format(str(len(info['videos'][0]['insights']['faces'][0]['thumbnails']))))
    info['videos'][0]['insights']['faces'][0]['thumbnails']


# In[13]:


images = []
img_raw = []
img_strs = []
for each_thumb in info['videos'][0]['insights']['faces'][0]['thumbnails']:
    if 'fileName' in each_thumb and 'id' in each_thumb:
        file_name = each_thumb['fileName']
        thumb_id = each_thumb['id']
        img_code = video_analysis.get_thumbnail_from_video_indexer(video_id,  thumb_id)
        img_strs.append(img_code)
        img_stream = io.BytesIO(img_code)
        img_raw.append(img_stream)
        img = Image.open(img_stream)
        images.append(img)


# In[16]:


for img in images:
    print(img.info)
    plt.figure()
    plt.imshow(img)


# In[ ]:





# In[ ]:




