import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5,6,7,9])
y = x*2+10

print("x is : ",x)
print("y is : ",y)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1)
])
model.compile()
model.fit(x,y,epochs=10)

print("Prediction : ",(model.predict((np.array([10,20,30,80])-mean_x)/std_x) * std_y)+mean_y)
