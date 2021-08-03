import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.Sequential()  # feed forward
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# it doesn't try to maximize the accuracy but it tries to minimize the loss
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1)

plt.imshow(x_test[0], cmap=plt.cm.binary)  # optional binary
plt.show()

model.save('Mnist_Classifier.model')

new_model = tf.keras.models.load_model('Mnist_Classifier.model')
predictions = new_model.predict(x_test)

print(np.argmax(predictions[0]))
