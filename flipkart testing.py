
# coding: utf-8

# In[1]:


import csv
import math

from PIL import Image
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Conv2D, Reshape
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import epsilon
import pandas as pd


# In[3]:


def create_model(trainable=False):

    model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, alpha=1.0)

    # to freeze layers
    for layer in model.layers:
        layer.trainable = trainable

    x = model.layers[-1].output
    x = Conv2D(4, kernel_size=3, name="coords")(x)
    x = Reshape((4,))(x)

    return Model(inputs=model.input, outputs=x)


# In[5]:


def create_model_1(trainable=False):
    model = DenseNet169(input_shape=(96, 96, 3), include_top=False)

    
    for layer in model.layers:
        layer.trainable = trainable

    x = model.layers[-1].output
    x = Conv2D(4, kernel_size=3, name="coords")(x)
    x = Reshape((4,))(x)

    return Model(inputs=model.input, outputs=x)


# In[4]:


model=create_model()
fname="model-flipkart-r3-4-cnn.h5"
model.load_weights(fname)


# In[5]:


import cv2
import matplotlib.pyplot as plt
import pickle


# In[6]:


pickle_in=open("flipkart_test_r3.pickle","rb")
img=pickle.load(pickle_in)


# In[7]:


plt.imshow(img[12814])
plt.show()


# In[8]:


df=cv2.imread("C:/Users/Avinash/Desktop/images/1458173279169DSC_0496.png")
df1=cv2.resize(df,(96,96))
plt.imshow(df1)
plt.show()
df1=np.array(df1)


# In[31]:


feat_scaled = preprocess_input(np.array(df1, dtype=np.float32))


# In[32]:


region = model.predict(x=np.array([feat_scaled]))[0]

x0 = int(region[0] * 640 / 96)
y0 = int(region[1] * 480 / 96)

x1 = int((region[0] + region[2]) * 640 / 96)
y1 = int((region[1] + region[3]) * 480 / 96)


# In[12]:


x0,y0,x1,y1


# In[14]:


img=cv2.resize(img[5677],(640,480))


# In[16]:


df1=cv2.rectangle(df, (x0, y0), (x1, y1), (0, 255, 255), 1)


# In[17]:


plt.imshow(df1)
plt.show()


# In[9]:


test=pd.read_csv("C:/Users/Avinash/Downloads/test_r3.csv")


# In[10]:


from tqdm import tqdm


# In[11]:


len(test)


# In[12]:


x11=[]
x00=[]
y11=[]
y00=[]


for i in tqdm(range(len(test))):
    img22=img[i]
    feat_scaled = preprocess_input(np.array(img22, dtype=np.float32))
    region = model.predict(x=np.array([feat_scaled]))[0]
    x0 = int(region[0] * 640 / 96)
    x00.append(x0)
    y0 = int(region[1] * 480 / 96)
    y00.append(y0)
    x1 = int((region[0] + region[2]) * 640 / 96)
    x11.append(x1)
    y1 = int((region[1] + region[3]) * 480 / 96)
    y11.append(y1)


# In[13]:


test.head()


# In[14]:


test['x1']=x00
test['x2']=x11
test['y1']=y00
test['y2']=y11


# In[15]:


test.head()


# In[16]:


pd.DataFrame(test, columns=['image_name','x1','x2','y1','y2']).to_csv('train_new_flipkart_r3_3.csv',index=False)


# In[17]:


f=pd.read_csv("C:/Users/Avinash/Downloads/train_new_flipkart_r3_3.csv")


# In[18]:


im=cv2.resize(img[1],(640,480))


# In[19]:


f.y2.iloc[1]


# In[23]:


i=6598

im=cv2.resize(img[i],(640,480))
x1=f.x1.iloc[i]
y1=f.y1.iloc[i]
x2=f.x2.iloc[i]
y2=f.y2.iloc[i]


im=cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),2)
plt.imshow(im)
plt.show()


# In[28]:




