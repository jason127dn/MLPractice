from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.datasets import mnist
from keras.utils import np_utils


model = load_model('my_model.h5')
(x_train, y_train),(x_test, y_test) = mnist.load_data()

y_Trainonehot = np_utils.to_categorical(y_train)
y_Testonehot = np_utils.to_categorical(y_test)

x_train_2D = x_train.reshape(60000, 28,28,1).astype('float32')
x_test_2D = x_test.reshape(10000, 28,28,1).astype('float32')

x_Train_norm = x_train_2D/255
x_Test_norm = x_test_2D/255
scores = model.evaluate(x_Test_norm, y_Testonehot)
print(scores)
