import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#load mnist dataset and split them into train and test 
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#creat sequential model
model = tf.keras.models.Sequential()  
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# we are not going to maximize the accuracy, we wanna tries to minimize the loss
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

#save your model, next time that want use it you can load model
model.save('Mnist_Classifier.model')

#load model
new_model = tf.keras.models.load_model('Mnist_Classifier.model')

#lets show one of test images and classify it
plt.imshow(x_test[0], cmap=plt.cm.binary)  
plt.show()

#make prediction
predictions = new_model.predict(x_test)

print(np.argmax(predictions[0]))
