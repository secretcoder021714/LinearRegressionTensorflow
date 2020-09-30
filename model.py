#Import section
import numpy as np
print("Version of np : " , np.__version__)
import tensorflow as tf
print("Version of tensorflow : " , tf.__version__)
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5,6,7,9])
y = x*2+10

#ploting for un-normalize data
plt.plot(x,y)
plt.show()

mean_x = x.mean()
x = x - mean_x
std_x = x.std()
x = x / std_x

mean_y = y.mean()
y = y - mean_y
std_y = y.std()
y = y / std_y

#ploting for normalize data
plt.plot(x,y)
plt.show()


print("x is : ",x)
print("y is : ",y)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1 , input_shape = [1,])
])
model.summary()
model.compile(optimizer = "rmsprop" , loss = "mae")
his = model.fit(x,y,epochs=10)

import matplotlib.pyplot as plt
plt.plot(his.history['loss'])
plt.title("Loss Curve")
plt.show()

print("Prediction : ",(model.predict((np.array([10,20,30,80])-mean_x)/std_x) * std_y)+mean_y)
