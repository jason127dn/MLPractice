from keras.layers import Dense, Input,Flatten,Reshape
from keras.models import Sequential, load_model, Model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test) = mnist.load_data()

x_train_2D = x_train.reshape(60000, 28,28,1).astype('float32')
x_test_2D = x_test.reshape(10000, 28,28,1).astype('float32')
x_train_norm=(x_train_2D/255.0)
x_test_norm=(x_test_2D/255.0)

input_img=Input(shape=(28,28,1),name="input")
x=Flatten()(input_img)
encoder=Dense(units=1000,activation='relu',name="d1")(x)
encoder=Dense(units=200,activation='relu',name="d2")(encoder)
encoder=Dense(units=50,activation='relu',name="d3")(encoder)
encoder=Dense(units=2,name='e_out')(encoder)
ecd=Model(input_img,encoder,name='encoder')
ecd.summary()

decoder_input=Input(shape=(2,),name='decoder_input')
decoder=Dense(units=50,activation='relu',name='d4')(decoder_input)
decoder=Dense(units=200,activation='relu',name='d5')(decoder)
decoder=Dense(units=1000,activation='relu',name='d6')(decoder)
decoder=Dense(units=784,activation='sigmoid',name='d_out')(decoder)
decoder=Reshape((28,28,1))(decoder)
dcd=Model(decoder_input,decoder,name='decoder')
autoencoder = Model(input_img,dcd(ecd(input_img)),name='autoencoder')

autoencoder.compile(loss='mean_squared_error',optimizer='adam')

train_history = autoencoder.fit(x_train_norm, x_train_norm, validation_split=0.2, epochs=30, batch_size=800, verbose=2)

autoencoder.save_weights("autoecd.weight")
autoencoder.save('autoecd.h5')

encoded_imgs = ecd.predict(x_test_norm)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()