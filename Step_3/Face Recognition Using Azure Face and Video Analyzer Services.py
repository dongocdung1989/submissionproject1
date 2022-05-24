#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


CONFIG = {
    'SUBSCRIPTION_KEY': '2e22306c33174c65821041bebacf5042',
    'LOCATION': 'Trial',
    'ACCOUNT_ID': '3d321732-f1e5-4517-a86d-515dfa1109f0'
}

video_analysis = VideoIndexer(
    vi_subscription_key=CONFIG['SUBSCRIPTION_KEY'],
    vi_location=CONFIG['LOCATION'],
    vi_account_id=CONFIG['ACCOUNT_ID']
)
video_analysis.check_access_token()

video_id = '7fc6b7ff52'
video_analysis.get_video_info(video_id)
info = video_analysis.get_video_info(video_id, video_language='English')
if len(info['videos'][0]['insights']['faces'][0]['thumbnails']):
    print("We found {} faces in this video.".format(str(len(info['videos'][0]['insights']['faces'][0]['thumbnails']))))
    
info['videos'][0]['insights']['faces'][0]['thumbnails']


# In[3]:


if len(info['videos'][0]['insights']['faces'][0]['thumbnails']):
    print("We found {} faces in this video.".format(str(len(info['videos'][0]['insights']['faces'][0]['thumbnails']))))
    info['videos'][0]['insights']['faces'][0]['thumbnails']

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


# In[4]:


for img in images:
    print(img.info)
    plt.figure()
    plt.imshow(img)

i = 1
for img in images:
    print(type(img))
    img.save('human-face' + str(i) + '.jpg')
    i= i+ 1
get_ipython().system('ls human-face*.jpg')


# In[5]:


thumbnail_id='0970d741-ff98-4fd9-af49-e67c9924a482'
img_code = video_analysis.get_thumbnail_from_video_indexer(video_id,  thumbnail_id)
print(img_code)
img_code = video_analysis.get_thumbnail_from_video_indexer(video_id,  thumbnail_id)
img_stream = io.BytesIO(img_code)
img = Image.open(img_stream)
imshow(img)

keyframes = []
for shot in info["videos"][0]["insights"]["shots"]:
    for keyframe in shot["keyFrames"]:
        keyframes.append(keyframe["instances"][0]['thumbnailId'])
for keyframe in keyframes:
    img_str = video_analysis.get_thumbnail_from_video_indexer(video_id,  keyframe)
info['summarizedInsights']['sentiments']
info['summarizedInsights']['emotions']


# In[6]:


DUNGDO_FACE_KEY = "004c79af87ba4003a53d692a0dfc0f46"
DUNGDO_FACE_ENDPOINT = "https://dungdofacecognitiveservice.cognitiveservices.azure.com/"
face_client = FaceClient(DUNGDO_FACE_ENDPOINT, CognitiveServicesCredentials(DUNGDO_FACE_KEY))

face_client.api_version


# In[7]:


get_ipython().system('ls human-face*.jpg')
my_face_images = [file for file in glob.glob('*.jpg') if file.startswith("human-face")]
print(my_face_images)

for img in my_face_images:
    with open(img, 'rb') as img_code:
        img_view_ready = Image.open(img_code)
        plt.figure()
        plt.imshow(img_view_ready)
        
PERSON_GROUP_ID = str(uuid.uuid4())
person_group_name = 'person-dungdo'


# In[8]:


# ## This code is taken from Azure Face SDK 
# ## ---------------------------------------
# def build_person_group(client, person_group_id, pgp_name):
#     print('Create and build a person group...')
#     # Create empty Person Group. Person Group ID must be lower case, alphanumeric, and/or with '-', '_'.
#     print('Person group ID:', person_group_id)
#     client.person_group.create(person_group_id = person_group_id, name=person_group_id)

#     # Create a person group person.
#     human_person = client.person_group_person.create(person_group_id, pgp_name)
#     # Find all jpeg human images in working directory.
#     human_face_images = [file for file in glob.glob('*.jpg') if file.startswith("human-face")]
#     # Add images to a Person object
#     for image_p in human_face_images:
#         with open(image_p, 'rb') as w:
#             client.person_group_person.add_face_from_stream(person_group_id, human_person.person_id, w)

#     # Train the person group, after a Person object with many images were added to it.
#     client.person_group.train(person_group_id)

#     # Wait for training to finish.
#     while (True):
#         training_status = client.person_group.get_training_status(person_group_id)
#         print("Training status: {}.".format(training_status.status))
#         if (training_status.status is TrainingStatusType.succeeded):
#             break
#         elif (training_status.status is TrainingStatusType.failed):
#             client.person_group.delete(person_group_id=PERSON_GROUP_ID)
#             sys.exit('Training the person group has failed.')
#         time.sleep(5)


# In[9]:


# build_person_group(face_client, PERSON_GROUP_ID, person_group_name)


# In[10]:


'''
Detect all faces in query image list, then add their face IDs to a new list.
'''
def detect_faces(client, query_images_list):
    print('Detecting faces in query images list...')

    face_ids = {} # Keep track of the image ID and the related image in a dictionary
    for image_name in query_images_list:
        image = open(image_name, 'rb') # BufferedReader
        print("Opening image: ", image.name)
        time.sleep(5)

        # Detect the faces in the query images list one at a time, returns list[DetectedFace]
        faces = client.face.detect_with_stream(image)  

        # Add all detected face IDs to a list
        for face in faces:
            print('Face ID', face.face_id, 'found in image', os.path.splitext(image.name)[0]+'.jpg')
            # Add the ID to a dictionary with image name as a key.
            # This assumes there is only one face per image (since you can't have duplicate keys)
            face_ids[image.name] = face.face_id

    return face_ids


# In[11]:


test_images = [file for file in glob.glob('*.jpg') if file.startswith("human-face")]


# In[12]:


test_images


# In[13]:


ids = detect_faces(face_client, test_images)


# In[14]:


ids


# In[15]:


# Verification example for faces of the same person.
verify_result = face_client.face.verify_face_to_face(ids['human-face1.jpg'], ids['human-face3.jpg'])
if verify_result.is_identical:
    print("Faces are of the same (Positive) person, similarity confidence: {}.".format(verify_result.confidence))
else:
    print("Faces are of different (Negative) persons, similarity confidence: {}.".format(verify_result.confidence))


# In[16]:


def show_image_in_cell(face_url):
    response = requests.get(face_url)
    img = Image.open(BytesIO(response.content))
    plt.figure(figsize=(10,5))
    plt.imshow(img)
    plt.show()


# In[17]:


dl_source_url = "https://raw.githubusercontent.com/dongocdung1989/Project1AutoKios/main/digital_id_DungDo.PNG"
show_image_in_cell(dl_source_url)


# In[18]:


dl_faces = face_client.face.detect_with_url(dl_source_url) 


# In[19]:


for face in dl_faces:
    print('Face ID', face.face_id, 'found in image', dl_source_url)
    # Add the ID to a dictionary with image name as a key.
    # This assumes there is only one face per image (since you can't have duplicate keys)
    ids['digital_id_DungDo.PNG'] = face.face_id


# In[20]:


ids


# In[21]:


# Verification example for faces of the same person.
dl_verify_result = face_client.face.verify_face_to_face(ids['human-face4.jpg'], ids['digital_id_DungDo.PNG'])


# In[22]:


if dl_verify_result.is_identical:
    print("Faces are of the same (Positive) person, similarity confidence: {}.".format(dl_verify_result.confidence))
else:
    print("Faces are of different (Negative) persons, similarity confidence: {}.".format(dl_verify_result.confidence))


# In[23]:


ids.values()


# In[24]:


dl_faces[0].face_rectangle.as_dict()


# In[25]:


# TAKEN FROM THE Azure SDK Sample
# Convert width height to a point in a rectangle
def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    
    return ((left, top), (right, bottom))


# In[26]:


def drawFaceRectangles(source_file, detected_face_object) :
    # Download the image from the url
    response = requests.get(source_file)
    img = Image.open(BytesIO(response.content))
    # Draw a red box around every detected faces
    draw = ImageDraw.Draw(img)
    for face in detected_face_object:
        draw.rectangle(getRectangle(face), outline='red', width = 10)
    return img


# In[27]:


drawFaceRectangles(dl_source_url, dl_faces)


# In[28]:


# A list of Face ID
ids


# In[29]:


# Enter the face ID of ca-dl-sample.png from the output of the cell above
get_the_face_id_from_the_driving_license = '0406daba-55f8-4544-b701-cef749101cbd'


# In[30]:


# person_gp_results = face_client.face.identify([get_the_face_id_from_the_driving_license], PERSON_GROUP_ID)


# In[31]:


# for result in person_gp_results:
#     for candidate in result.candidates:
#         print("The Identity match confidence is {}".format(candidate.confidence))


# In[ ]:




