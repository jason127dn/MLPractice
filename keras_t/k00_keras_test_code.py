from keras.layers import Dense, Input, Lambda
from keras.models import load_model, Model, Sequential
i1=Input(shape=(1,))
d11=Dense(units=100)
i1d1=d1(i1)
i2d1=d1(i2)
i3=Input(shape=(100,))
d3=Dense(units=200)
d4=Dense(units=200)
i3d3=d3(i3)
d3d4=d4(i3d3)

from keras.layers import Dense, Input, Lambda
from keras.models import load_model, Model, Sequential

i1=Input(shape=(1,))
x=Dense(10,name='d11')(i1)
x=Dense(10,name='d12')(x)
x=Dense(10,name='d13')(x)
x=Dense(10,name='d14')(x)
x=Dense(10,name='d15')(x)
o1=Dense(100)(x)
m1=Model(i1,o1)

i2=Input(shape=(100,))
y=Dense(200,name='d21')(i2)
y=Dense(200,name='d22')(y)
y=Dense(200,name='d23')(y)
y=Dense(200,name='d24')(y)
y=Dense(200,name='d25')(y)
o2=Dense(300)(y)

m2=Model(i2,o2)
m3=Model(i1,m2(m1(i1)))

d=m2.layers[3](m2.layers[2](m2.layers[1](m1.get_output_at(1))))
M4=Model(i1,d)

z=Dense(300,name='d31')
z=Dense(200,name='d32')(z)
z=Dense(200,name='d33')(z)
z=Dense(200,name='d34')(z)
z=Dense(200,name='d35')(z)
o3=Dense(300)(z)

m2=Model(i2,o2)
m3=Model(i1,m2(m1(i1)))
m4=Model(i1,m1.layers[4].output)

d=m2.layers[3](m2.layers[2](m2.layers[1](m1.get_output_at(1))))
M4=Model(i1,d)