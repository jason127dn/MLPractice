from keras.layers import Dense, Input
from keras.models import Sequential, load_model, Model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test) = mnist.load_data()

x_train_2D = x_train.reshape(60000, 28*28).astype('float32')
x_test_2D = x_test.reshape(10000, 28*28).astype('float32')
x_train_norm=(x_train_2D/255.0)
x_test_norm=(x_test_2D/255.0)

input_img=Input(shape=(784,),name="input")
encoder=Dense(units=1000,activation='relu',name="d1")(input_img)
encoder=Dense(units=200,activation='relu',name="d2")(encoder)
encoder=Dense(units=50,activation='relu',name="d3")(encoder)
encoder_output=Dense(units=2,name='e_out')(encoder)

decoder=Dense(units=50,activation='relu',name='d4')(encoder_output)
decoder=Dense(units=200,activation='relu',name='d5')(decoder)
decoder=Dense(units=1000,activation='relu',name='d6')(decoder)
decoder=Dense(units=784,activation='sigmoid',name='d_out')(decoder)

autoencoder = Model(input=input_img, output=decoder)
encode = Model(input=input_img, output=encoder_output)

autoencoder.compile(loss='mean_squared_error',optimizer='adam')

train_history = autoencoder.fit(x=x_train_norm, y=x_train_norm, validation_split=0.2, epochs=30, batch_size=800, verbose=2)

autoencoder.save_weights("autoecd.weight")
autoencoder.save('autoecd.h5')

encoded_imgs = encode.predict(x_test_norm)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()