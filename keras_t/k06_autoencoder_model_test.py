from keras.layers import Dense, Input
from keras.models import Sequential, load_model, Model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
model_name=['models/autoencodercnn.h5','autoecd.h5']
image_size=28
(x_train, y_train),(x_test,y_test) = mnist.load_data()

x_train_2D = x_train.reshape(60000, 28,28,1).astype('float32')
x_test_2D = x_test.reshape(10000, 28,28,1).astype('float32')
x_train_norm=(x_train_2D/255.0)
x_test_norm=(x_test_2D/255.0)

autoencoder = load_model(model_name[1])

#encoder = Model(inputs=autoencoder.input,outputs=autoencoder.get_layer("encoder").output)
#encoder = Model(inputs=autoencoder.layers[0].input,outputs=autoencoder.layers[5].output)
encoder=autoencoder.get_layer('encoder')
encoded_imgs=encoder.predict(x_test_norm)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()

# Predict the Autoencoder output from corrupted test images
x_decoded = autoencoder.predict(x_test_norm)


# Display the 1st 8 corrupted and denoised images
rows, cols = 10, 30
num = rows * cols
imgs = np.concatenate([x_test_norm[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 2, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 2, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          
          'autoencoder images:  second rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()


intputcode=np.empty((30,30,2))
_x=np.arange(-20,10,1)
_y=np.arange(0,30,1)

_x,_y=np.meshgrid(_x,_y)
intputcode[:,:,0]=_x
intputcode[:,:,1]=_y
intputcode=intputcode.reshape((900,2))
decoder=autoencoder.get_layer('decoder')
g_img=decoder.predict(intputcode)
print(g_img.shape)
# Display the 1st 8 corrupted and denoised images
rows, cols = 30, 30
num = rows * cols

imgs = g_img.reshape((rows, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = g_img.reshape((rows , -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '

          'autoencoder images:  second rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()

