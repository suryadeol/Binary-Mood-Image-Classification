#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import matplotlib.pyplot as plt


# In[2]:


#s=os.path.join('E:\Image classification\Data')
#s


# In[3]:


#Out of memory error avoidance
ngpus=tf.config.experimental.list_physical_devices('GPU')
for g in ngpus:
    tf.config.experimental.set_memory_growth(gpu,True)



# In[4]:


#PATH ALLOCATION
data_dir=os.path.join('data')
classes=os.listdir(data_dir)
classes
#data.class_names


# In[5]:


#REMOVING DOGGY IMAGES
import cv2
import imghdr                                                                   #you can read image using either cv2 or matplotlib
                                                                                #cv2 uses BGR,matplotlib uses RGB
img_ext=['jpeg','jpg','png','bmp']                                              #if we use cv2,then convert imge to RGB to display

for img_c in classes:
    for img_v in os.listdir(os.path.join('data',img_c)):
        img_p=os.path.join('data',img_c,img_v)
        try:
            img_mat=cv2.imread(img_p)
            #plt.imread(img_p)
            #plt.imshow(k #an matrix of image)
            #plt.imshow(img_mat)     #gives BGR cv2 image
            img_tip=imghdr.what(img_p)
            if(img_tip not in img_ext):
                print('Image not in ext list {}'.format(img_p))
                os.remove(img_p)
            
        except Exception as e:
            #print('Issue with image {}'.format(img_p))
            print(e)
                


# In[6]:


#image.shape----->gives the shape of the image with  channels
#plt.imshow(img_mat) #BGR formate
#plt.imshow(cv2.cvtColor(img_mat,cv2.COLOR_BGR2RGB)) #BGR TO RGB


# In[7]:


#LOADING DATA
from numpy import *
#tf.keras.utils.image_dataset_from_directory??


# In[8]:


data=tf.keras.utils.image_dataset_from_directory('data')
#data=tf.keras.utils.image_dataset_from_directory('data',batch_size=8,image_size=(128,128))
data


# In[9]:


data_iter=data.as_numpy_iterator()  #used to loop the data pipeline 
data_iter


# In[10]:


batch=data_iter.next()  #pulls the data batches from pipeline through a tupple
#batch
#batch[0].shape         #batch is tupple with 0 index contains images in that batch
                        #1 index contain label
batch[1]                #labels of classes   0-->happy 1-->sad


# In[11]:


'''for i,j in data.take(1):
    print(i.shape)         #i holdes images j hold lables
    print(j.numpy)'''     #same as iterator--->loops the batch as required using take method


# In[12]:


#plot sample data
fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for i,j in enumerate(batch[0][:4]):
    ax[i].imshow(j.astype(int))
    ax[i].title.set_text(batch[1][i])


# In[13]:


#--------PREPROCESSING DATA----------#

#optimizing the RGB values
#nbatch=batch[0]/255
#optimizing at the time of pipeline
data=data.map(lambda x,y:(x/255,y))    #x independent features,y dependent feature
scaled=data.as_numpy_iterator()
batch=scaled.next()


fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for i,j in enumerate(batch[0][:4]):
    ax[i].imshow(j)    #already in minimal scaled range
    ax[i].title.set_text(batch[1][i])


# In[14]:


#splitting the dataSet
print(len(data))
train=int(0.7*len(data)) #train--->70%
val=int(0.2*len(data))  #test---->20%
test=int(0.1*len(data))   #validation----->10%
print(train,test,val)

#shuffle dataset
data=data.shuffle(10000,seed=12)

#taking data
train_ds=data.take(train)
val_ds=data.skip(train).take(val)
test_ds=data.skip(test+val).take(test)


# In[15]:


#MODEL BULIDING
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dense,Flatten,Dropout


# In[16]:


model=Sequential()


# In[17]:


model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[18]:


model.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])


# In[19]:


model.summary()

#maxpool---->2*2-----> ex:254/2=127
#dense layer--->gives single output
#param--->neruons


# In[20]:


#TRAINING
tensor_call=tf.keras.callbacks.TensorBoard(log_dir='logs')
tensor_call
#using logs & call backs we can manage the learning phase at any instance


# In[21]:


#fit the model---->training component
history=model.fit(train_ds,epochs=10,validation_data=val_ds,callbacks=[tensor_call])


# In[22]:


di=history.history #gives entire trained DL model info like loss,validation loss and accuracy
di


# In[24]:


#<----PERFORMANCE------>

#LOSS VIZULATIZATION
plt.figure(figsize=(5,5))
plt.plot(di['loss'],color="red",label="loss")
plt.plot(di['val_loss'],color="green",label="val_loss")
plt.show()
#observe by considering Y axis for both--->graphs down means accuacry increasing


# In[27]:


#ACCURACY VIZULIZATION
plt.figure(figsize=(5,5))
plt.plot(di['accuracy'],color="red",label="loss")
plt.plot(di['val_accuracy'],color="green",label="val_loss")
plt.show()


# In[28]:


#evalute
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test_ds.as_numpy_iterator(): 
    X, y = batch     #X->holds images #y->holds lables
    res = model.predict(X)  #32 predictions for test data
    pre.update_state(y, res)
    re.update_state(y, res)
    acc.update_state(y, res)
print(pre.result(), re.result(), acc.result())


# In[29]:


#unseen prediction
img=plt.imread('my.jpg')
plt.imshow(img)

#resize according to layers
resize=tf.image.resize(img,(256,256))
plt.imshow(resize.numpy().astype(int)) #adjusted according to RGB b/w 0 to 1
plt.show()

#optimize the new image
resize=resize/255
#expand your image array
img=expand_dims(resize,0)


# In[30]:


#prediction
pred=model.predict(img)
print(pred)


# In[1]:


#since it is a binary classification

if pred[0]>0.5:
    print("SAD")
else:
    print("Happy")


# In[32]:


from tensorflow.keras.models import load_model


# In[34]:


model.save(os.path.join('image classification.h5'))


# In[35]:


#we are loading the model to make predictions from already existed model
#loading model will solve re execution of code
#if we want to run entire code again we can but every new run will provide model by saving and we can use by loading
#new_model=load_model('image classification.h5')  #os.join.path('image classification')


# In[36]:


#loaded model prediticion
#new_model.predict(img)


# In[ ]:




