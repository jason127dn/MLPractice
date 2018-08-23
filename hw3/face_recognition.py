from keras import backend as K
from keras.models import load_model,Model
from keras.layers import Dropout, Flatten,Dense,Conv2D,Reshape,Input,Activation,MaxPooling2D
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import csv


label = []
feature = []
i=0

def get_img(st):
    img=[]
    px=st.split(' ')
    for p in px:
        img.append(float(p)/255.)
    return np.array(img).reshape((48,48,1))


with open('train.csv',newline='') as csvfile:
    csvfile.readline()
    rows=csv.reader(csvfile)

    for row in rows:
        label.append(row[0])
        feature.append(get_img(row[1]))
        print(i)
        i = i+1


feature = np.array(feature)
label = np.array(label)
labelc=np_utils.to_categorical(label)

x_train = feature[:20000]
x_test = feature[20000:]
y_train = labelc[:20000]
y_test = labelc[20000:]



input_layer=Input((48,48,1),name='intput_layer')
x=Conv2D(filters=20,kernel_size=(5,5),name='conv1',padding='same',activation='relu')(input_layer)
#x=Conv2D(filters=30,kernel_size=(5,5),name='conv2',padding='same',activation='relu')(x)
x=MaxPooling2D(pool_size=(3,3),name='p1')(x)
x=Conv2D(filters=50,kernel_size=(3,3),name='conv3',padding='same',activation='relu')(x)
#x=Conv2D(filters=30,kernel_size=(3,3),name='conv4',padding='same',activation='relu')(x)
x=MaxPooling2D(pool_size=(3,3),name='p2')(x)
x=Flatten(name='f1')(x)
x=Dense(units=1000,kernel_initializer='normal',activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(units=300,kernel_initializer='normal',activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(units=7,kernel_initializer='normal',activation='softmax')(x)

model=Model(input_layer,x)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=x_train, y=y_train, validation_split=0.33, epochs=20, batch_size=400)


