#[V] add layer before load model
#[ ] set untrainable layer
#[ ] get output from nn
#[ ] custom loss function

from keras.models import load_model,Model
from keras.layers import Dense,Conv2D,Reshape,Input,Activation
import numpy as np

model_path='models/cnn.h5'

p1_f=np.zero((250000,14,14,30))

cnn_model=load_model(model_path)
input_data=np.ones((250000,1))
cnn_model.trainable=False
i_layer=Input(shape=(1,))
x=Dense(units=28*28)(i_layer)
x=Activation('sigmoid',name='sa')(x)
x=Reshape((28,28,1))(x)
head=Model(i_layer,x,name="head")
head.summary()
all_m=Model(i_layer,cnn_model(head(i_layer)))
all_m.summary()
visualizer=Model(i_layer,all_m.layers[2].layers[1].output)


def vf_loss(intput ,output):

print("1")
