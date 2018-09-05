#machine learning note

learning resource

* [Hung-yi lee](http://speech.ee.ntu.edu.tw/~tlkagk/courses.html),[youtube channel](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ)
* [morvanzhou](https://morvanzhou.github.io/)
* [keras](https://keras.io/zh/)

## 0.Start with Python coding
* tutorial

[basic python skill](https://www.youtube.com/playlist?list=PL6gx4Cwl9DGAcbMi1sH6oAMk4JHw91mC_)

* IDE 

[pycharm](https://www.jetbrains.com/pycharm/)

 
## 1.[Introduction to machine learning](https://www.youtube.com/watch?v=CXgbekl66jc)

#### 1.1 what is machine learning 
finding a function 

example:
* speech recognition    sound -> text
* image recognition     image -> object

what can we do 

* prediction
* classification
* ...

#### 1.2 how did machine learn
problem we will solute
1. prodiction
2. classification
    1) binary
    2) multi-class
3.ine learning work

three step:

* Define a set of function  
    seting your model(linear,nonlinear,deep)

* Goodness of function f  
    define loss function

* Pick the best one  
    make loss function as small as you can
    

## 2 regression
output a scalar  

#### 2.1
three step:
1. Model(linear)  
y=b+wx

2. loss function  
    main-squire-error  
    L(f)=L(w,b)=sum((y<sub>h</sub>-(wx+b))<sup>2</sup>)

    how to define loss function -> [loss](https://keras.io/zh/losses/)
        
      1. mean-squared-error
      2. categorial_crossentropy
      99. ...  

3. gradient descent  
    w <sub>n</sub> =w<sub>n-1</sub>- r * dL/dw

    how to adaptive learning rate -> [optimizers](https://keras.io/zh/optimizers/) in keras
        
      1. adaptive
      2. Adagrad
      3. SGD
      4. RMSprop
      5. Adadelta
      6. Adam
      99. ...
    tips  
    1. Stochastic Gradient Descent
    2. Feature Scaling
    

problem:  
overfitting underfitting(how to find best model)

#### 2.2
where do the error come from
1. bias
    underfitting  
    because your model don't cover the right 
    function 
2. Variance
    overfitting  
    model has too many choose that the final picked 
    function will only fit the training data

use cross validation

## 3.Classification
#### 3.1 logistic regression
1. model  
sigmiod function: f<sub>w,b1</sub>(x)=1/(1+e<sup>-(wx+b)</sup>) 
[activation]() in keras
2. loss  
cross entropy: L(f)=sum(C(f(x<sup>n</sup>),y<sup>n</sup>))  
3. gradient descent
#### 3.2 multiclass regression
1. model 
use softmax:

3.3 limitation  
make it deep!!

## 4. Deep learning 
#### 4.1 Nerual Network
1. model  
full-connected([desence]()),cnn([conv2D]),rnn([LSTM]),....
2. loss  
forward pass
3. gradient descent  
backpropagation
#### 4.2 tips
problem training bad or testing bad  
training bad:  
deep is not always good : gradient vanishing  
* new activation function :Relu (fast, ),Maxout
* RMSProp, Momentum, Adam  
testing bad(overfitting):
* regularization
* dropout

## 5. Keras


## 6. Convolutional Netual Network
#### 6.1 convolution

#### 6.2 maxpooling
#### 6.3 cnn in keras
#### 6.4 application
1. deep dream
2. deepstyle
3. alphaGo
4. speech
####　6.5 training tips
1. 

## 7 RNN
## 8 GANs
## 9 Reinforce Learning








    
