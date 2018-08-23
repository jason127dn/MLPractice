#[V] add layer before load model
#[V] set untrainable layer
#[V] get output from nn
#[V] custom loss function
from keras import backend as K
from keras.models import load_model,Model
from keras.layers import Dense,Conv2D,Reshape,Input,Activation
import numpy as np
import matplotlib.pyplot as plt

model_path='models/cnn.h5'
batch_size=100

training_data=10000
p1_f=np.zeros((training_data,14,14,30))
p1_f[:,:,:,0]=-1

input_data=np.ones((training_data,1))

cnn_model=load_model(model_path)
cnn_model.trainable=False
i_layer=Input(shape=(1,))
x=Dense(units=28*28)(i_layer)
x=Dense(units=28*28)(x)
x=Activation('tanh',name='sa')(x)
x=Reshape((28,28,1))(x)
head=Model(i_layer,x,name="head")
head.summary()

sub_cnn_model=Model(cnn_model.layers[0].input,cnn_model.layers[2].output)
sub_cnn_model.trainable=False
visualizer=Model(i_layer,sub_cnn_model(head.get_output_at(0)))
imgs=[]
plt.figure()
for i in range(3):
    def fv_loss(y_true, y_pred):
        return K.sum(-1*y_pred[:,:,:,i])


    visualizer.compile(optimizer='adam',loss=fv_loss)
    visualizer.fit(input_data,p1_f,epochs=4,batch_size=batch_size)

    img=head.predict([1.])

    plt.subplot(6, 8, i+1)
    plt.title(i)
    plt.axis('off')
    img = img.reshape((28, 28))
    img = ((img+1) * 255).astype(np.uint8)
    plt.imshow(img, interpolation='none', cmap='gray')

    print(i)
plt.show()