import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout,Activation,Reshape, Flatten , Conv2D , MaxPooling2D, Dropout
from keras.models import Model
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.models import model_from_json


(x_train, y_train),(x_test, y_test) = mnist.load_data()

'''
# model is sequential
model = Sequential()
model.add(Conv2D(filters=30,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu',name='c1'))
model.add(MaxPooling2D(pool_size=(2,2),name='p1'))
model.add(Conv2D(filters=40,kernel_size=(5,5),padding='same',activation='relu',name='c2'))
model.add(MaxPooling2D(pool_size=(2,2),name='p2'))
model.add(Flatten())
model.add(Dense(units=300, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=300, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=10, kernel_initializer='normal',activation='softmax'))
'''
#
input_layer=Input((28,28,1),name='inlayer')
x=input_layer
x=Conv2D(filters=30,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu',name='c1')(x)
x=MaxPooling2D(pool_size=(2,2),name='p1')(x)
x=Conv2D(filters=40,kernel_size=(5,5),padding='same',activation='relu',name='c2')(x)
x=MaxPooling2D(pool_size=(2,2),name='p2')(x)
x=Flatten()(x)
x=Dense(units=300, kernel_initializer='normal', activation='relu')(x)
x=Dropout(0.25)(x)
x=Dense(units=300, kernel_initializer='normal', activation='relu')(x)
x=Dense(units=10, kernel_initializer='normal',activation='softmax')(x)
model=Model(input_layer,x)
model.summary()
#
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
'''
i_layer=Input(shape=(1,))
x=Dense(units=28*28)(i_layer)
x=Activation('sigmoid',name='sa')(x)
x=Reshape((28,28,1))(x)
head=Model(i_layer,x,name="head")
head.summary()
all_m=Model(i_layer,model(head(i_layer)))
all_m.summary()
'''
y_Trainonehot = np_utils.to_categorical(y_train)
y_Testonehot = np_utils.to_categorical(y_test)

x_train_2D = x_train.reshape(60000, 28,28,1).astype('float32')
x_test_2D = x_test.reshape(10000, 28,28,1).astype('float32')

x_Train_norm = x_train_2D/255
x_Test_norm = x_test_2D/255

# 進行訓練, 訓練過程會存在 train_history 變數中
train_history = model.fit(x=x_Train_norm, y=y_Trainonehot, validation_split=0.33, epochs=20, batch_size=400, verbose=2)

# 顯示訓練成果(分數)
scores = model.evaluate(x_Test_norm, y_Testonehot)
print()
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))


model.save_weights("models/cnn.weight")
model.save('models/cnn.h5')

# 預測(prediction)
X = x_Test_norm[0:10,:]
predictions = model.predict_classes(X)
# get prediction result
print(predictions)

# 顯示 第一筆訓練資料的圖形，確認是否正確
plt.imshow(x_test[0])
plt.show()

plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Train History')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()

