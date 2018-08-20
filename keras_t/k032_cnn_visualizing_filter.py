from keras.models import load_model,Model
from keras.layers import Dense,Conv2D,Reshape,Input,Activation
import numpy as np

model_path='my_model.h5'
cnn_model=load_model(model_path)
input_data=np.ones((250000,1))
f_layer=Input(shape=(1,))
i_layer=Dense(units=28*28)(f_layer)
i_layer=Activation('sigmoid',name='sa')(i_layer)
i_layer=Reshape((28,28,1))(i_layer)
head=Model(f_layer,i_layer,name="head")
head.summary()
cnn_model.summary()
cnn_model.input=head.output
visualizer=Model(f_layer,cnn_model,name='first_visualizer')
print("1")
