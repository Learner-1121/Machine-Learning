

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
#sklearn
from sklearn.utils import shuffle"""
import cv2
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

import os

import eel

from PIL import Image
# SKLEARN
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
#%%
path1='C:/Users/Vennela/Desktop/Morning_resized'
path2='C:/Users/Vennela/Desktop/Evening_resized'
eel.init('WD')
#%%
listing_e=os.listdir(path2)
print(len(listing_e))
listing_m=os.listdir(path1)
listing_e.extend(listing_m)
num_samples=len(listing_e)
print(num_samples)
#%%

img1 = cv2.imread('C:/Users/Vennela/Desktop/Morning_resized/m1.jpg')
img1 = img1.flatten()
#%%
# create matrix to store all flattened images
immatrix = np.array([np.array(Image.open('C:/Users/Vennela/Desktop/Total/'+im2)).flatten() for im2 in listing_e])
new = np.array(immatrix).flatten().tolist()
#x_data = np.array( [np.array(cv2.imread('C:/Users/Vennela/Desktop/Total/'+listing_m[i]) for i in range(len(listing_m))] )
#
#pixels = x_data.flatten()

#%%
#create matrix to store all flattened images
#immatrix = np.array(np.array(Image.open('t'+im2)).flatten() for im2 in imlist)
#%%
list_total = os.listdir("C://Users/Vennela/Desktop/Total")
img_rows = 100
img_cols = 100
label=np.ones((num_samples,),dtype = int)

#%%
label[0:85]=0
        
#%%       

target_names = ['normal','tired']
data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]
#%%
img=immatrix[2].reshape(100,100)
plt.imshow(img,cmap='Greys_r')
#%%
batch_size = 8

nb_classes = 2
# number of epochs to train
nb_epoch = 10
# number of convolutional filters to use
nb_filters =128
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
(X, Y) = (train_data[0],train_data[1])
# STEP 1: split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
#%%
#img = X_train[0].reshape(200,200)
#print(img)
#%%
X_train = X_train.reshape(X_train.shape[0], 1,img_rows,img_cols)
#%%
X_test = X_test.reshape(X_test.shape[0], 1, 200,200)

#%%
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
# convert class vectors to categorical class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 51
plt.imshow(X_train[i, 0], interpolation='nearest',cmap='Greys_r')
print("label : ", Y_train[i,:])

#%%
model = Sequential()

model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='tanh',input_shape=(1, img_rows, img_cols), padding="valid",data_format='channels_first'))

model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))

model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='hinge', optimizer='sgd', metrics=["acc"])

#%%
from keras.callbacks import History 
history = History()
hist=model.fit(X_train,Y_train, batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,
               validation_data=(X_test, Y_test),callbacks=[history])

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.grid(True)
plt.xlabel('num of Epochs')
plt.ylabel('loss/accuracy')
plt.legend(['train_loss','val_loss','train_accu','val_accu'],loc=3)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#%%
score = model.evaluate(X_test, Y_test,  verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#%%
input_image=X_train[0:1,:,:,:]
print(input_image.shape)

plt.imshow(input_image[0,0,:,:],cmap ='gray')

from sklearn.metrics import classification_report,confusion_matrix

y_pred = model.predict_classes(X_test)

print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))



