from keras import backend as K
from keras.models import load_model,Model
from keras.layers import Dense,Conv2D,Reshape,Input,Activation
import numpy as np
import matplotlib.pyplot as plt
import csv


label = []
feature = []
i=0

def get_img(st):
    img=[]
    px=st.split(' ')
    for p in px:
        img.append(int(p))
    return np.array(img).reshape((48,48))


with open('train.csv',newline='') as csvfile:
    csvfile.readline()
    rows=csv.reader(csvfile)

    for row in rows:
        label.append(row[0])
        feature.append(get_img(row[1]))
        print(i)
        i = i+1


feature = np.array(feature)
label = np.array(label)
