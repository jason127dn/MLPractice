#[V] add layer before load model
#[ ] set untrainable layer
#[ ] get output from nn
#[ ] custom loss function

from keras.models import load_model,Model
from keras.layers import Dense,Conv2D,Reshape,Input,Activation
import numpy as np

model_path='keras_t/models/cnn.h5'

training_data=25000
p1_f=np.zeros((14,14,30))
p1_f[:,:,0]=-1

input_data=np.ones((training_data,1))

cnn_model=load_model(model_path)
cnn_model.trainable=False
i_layer=Input(shape=(1,))
x=Dense(units=28*28)(i_layer)
x=Activation('sigmoid',name='sa')(x)
x=Reshape((28,28,1))(x)
head=Model(i_layer,x,name="head")
head.summary()
all_m=Model(i_layer,cnn_model(head(i_layer)))
all_m.summary()
fv=cnn_model.layers[2](cnn_model.layers[1](head.get_output_at(0)))
visualizer=Model(i_layer,fv)

def fv_loss(intput ,output):
    return np.dot(fv,p1_f)


visualizer.compile(optimizer='adam',loss=fv_loss)

print("1")
