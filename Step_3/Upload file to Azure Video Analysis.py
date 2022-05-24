#!/usr/bin/env python
# coding: utf-8

# ### Please install the required Python modules/SDKs

# In[3]:


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


# In[4]:


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


# In[7]:


uploaded_video_id = video_analysis.upload_to_video_indexer(
   input_filename='/home/workspace/Dungdo-30s-video.mp4',
   video_name='Dungdo-30s-video',  # unique identifier for video in Video Indexer platform
   video_language='English'
)

info = video_analysis.get_video_info(uploaded_video_id, video_language='English')


# In[ ]:





# In[ ]:




