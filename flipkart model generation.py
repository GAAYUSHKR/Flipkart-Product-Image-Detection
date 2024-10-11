
# coding: utf-8

# In[1]:


import csv
import math

from PIL import Image
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Conv2D, Reshape,UpSampling2D, Reshape, BatchNormalization, Activation,Concatenate
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import epsilon

ALPHA = 1.0

IMAGE_SIZE = 96

image_width=640
image_height=480


BATCH_SIZE = 64


TRAIN_CSV = "C:/Users/Avinash/Downloads/training_set.csv"


# In[15]:


from tensorflow.keras.applications.densenet import DenseNet169


# In[2]:


import pandas as pd
df=pd.read_csv(TRAIN_CSV)
df.head()
df.shape


# In[13]:


for index,row in enumerate(df.values):
    print(row[0])
    print(index)


# In[3]:


df.isnull().count()


# In[5]:


class DataGenerator_2(Sequence):

    def __init__(self, csv_file):
        self.paths = []
        
        self.coords=np.zeros((24000,4))
        
        reader=pd.read_csv(TRAIN_CSV)
        for index, row in enumerate(reader.values):
            for i, r in enumerate(row[1:5]):
                row[i+1] = int(r)

            path,x0, x1, y0, y1= row
            self.coords[index, 0] = x0 * IMAGE_SIZE / image_width
            self.coords[index, 1] = y0 * IMAGE_SIZE / image_height
            self.coords[index, 2] = (x1 - x0) * IMAGE_SIZE / image_width
            self.coords[index, 3] = (y1 - y0) * IMAGE_SIZE / image_height 
            path="C:/Users/Avinash/Downloads/flipkart_images/images/"+path
            self.paths.append(path)

     
    def __len__(self):
        return math.ceil(len(self.coords) / BATCH_SIZE)

    def __getitem__(self, idx):
        try:
            
            batch_paths = self.paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
            batch_coords = self.coords[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        
            #print(batch_coords)
            batch_images = np.zeros((len(batch_paths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
            for i, f in enumerate(batch_paths):
                img = Image.open(f)
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img = img.convert('RGB')

                batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))
                img.close()
        except Exception as e:
            pass 
        return batch_images, batch_coords


# In[6]:


d = DataGenerator_2(TRAIN_CSV)
print(d.__getitem__(0))


# In[7]:


def create_model(trainable=False):
    model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, alpha=ALPHA)

    # to freeze layers
    for layer in model.layers:
        layer.trainable = trainable

    x = model.layers[-1].output
    x = Conv2D(4, kernel_size=3, name="coords")(x)
    #x = BatchNormalization()(x)
    #x = Activation("relu")(x)
    x = Reshape((4,))(x)

    return Model(inputs=model.input, outputs=x)


# In[8]:


model = create_model()
model.summary()

train_datagen = DataGenerator_2(TRAIN_CSV)

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])


# In[15]:


train_datagen


# In[10]:


model.summary()

model.fit_generator(generator=train_datagen,epochs=10,verbose=1,shuffle=True)


# In[11]:


fname="model-flipkart-r3-4-cnn.h5"
model.save_weights(fname)


# In[19]:


#model.save('flipkart.model')


# In[32]:


import cv2
import matplotlib.pyplot as plt
import pickle


# In[12]:


unscaled=cv2.imread("file:///C:/Users/Avinash/Downloads/images/images/1458173367140DSC_0492.png")


# In[33]:


pickle_in=open("flipkart_test.pickle","rb")
x=pickle.load(pickle_in)


# In[34]:


plt.imshow(x[5677])
plt.show()


# In[35]:


feat_scaled = preprocess_input(np.array(x[5677], dtype=np.float32))


# In[36]:


region = model.predict(x=np.array([feat_scaled]))[0]

x0 = int(region[0] * 640 / 96)
y0 = int(region[1] * 480 / 96)

x1 = int((region[0] + region[2]) * 640 / 96)
y1 = int((region[1] + region[3]) * 480 / 96)


# In[37]:


print(x0,x1,y0,y1)


# In[38]:


img=cv2.resize(x[5677],(640,480))


# In[16]:


cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
cv2.imshow("image", img)


# In[39]:


cv2.rectangle(img, (258, 126), (438, 438), (0, 0, 255), 1)
cv2.imshow("image", img)


# In[40]:


plt.imshow(img)
plt.show()

