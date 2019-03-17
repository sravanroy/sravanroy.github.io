---
title: "Predicting digits in MNIST database"
date: 2019-02-17
tags: [deep learning, keras, handwritten digits]
header:
  image: "/images/number_prediction/3_array.png"
excerpt: "Deep learning, keras, handwritten digits"
mathjax: "true"
---

## *Simple prediction of digits using Pandas and TensorFlow !*
----
* A basic model to predict the handwritten images in MNIST database
+ TensorFlow API is implemented to achieve this purpose


The required packages are imported into the notebook as shown-
```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
```
The MNIST dataset is loaded from Keras library and unpacked to training and testing variables 
```python
  mnist = tf.keras.datasets.mnist #data base containing hand-written digits 0-9

(x_train,y_train), (x_test,y_test) = mnist.load_data()
```
 A sample image of digit in the training dataset
```python
 plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show() 
```
![alt]({{ site.url }}{{ site.baseurl }}/images/number_prediction/download.png)

Choosing and training the model
```python
 # normalizing the data to make the neural ntwork easier to learn
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#choosing the sequential model
model = tf.keras.models.Sequential()

#defining the architecture of the model
model.add(tf.keras.layers.Flatten(input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#defining the parameters to train the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#training the model
model.fit(x_train,y_train,epochs=3)

Epoch 1/3
60000/60000 [==============================] - 7s 123us/step - loss: 0.2671 - acc: 0.9223
Epoch 2/3
60000/60000 [==============================] - 10s 165us/step - loss: 0.1088 - acc: 0.9668
Epoch 3/3
60000/60000 [==============================] - 10s 167us/step - loss: 0.0725 - acc: 0.9770
```
Calculating the validation loss 
```python
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

10000/10000 [==============================] - 0s 43us/step
0.1025834274851717 0.9694
```
Saving and loading the model 
```python
#saving the model
model.save("num_reader.model")
#load the model
new_model=tf.keras.models.load_model("num_reader.model")
```
Predicting digits using the model
```python
predictions = new_model.predict([x_test])
print(predictions)
#prediction for first element in x_test is
print(np.argmax(predictions[0]))
7
plt.imshow(x_test[0])
plt.show()
```
![alt]({{ site.url }}{{ site.baseurl }}/images/number_prediction/download7.png)

*This is a basic walkthrough for building and predicting images from a model*
*The parameters for fitting the model can be flexible, with a wide range of loss functions and optimizers to experiment with!* 
 



 

