from keras.models import load_model,Model
from keras.layers import Dense,Conv2D,Reshape,Input,Activation
import numpy as np

model_path='models/cnn.h5'


cnn_model=load_model(model_path)
input_data=np.ones((250000,1))

i_layer=Input(shape=(1,))
x=Dense(units=28*28)(i_layer)
x=Activation('sigmoid',name='sa')(x)
x=Reshape((28,28,1))(x)
head=Model(i_layer,x,name="head")
head.summary()
all_m=Model(i_layer,cnn_model(head(i_layer)))
all_m.summary()

print("1")
