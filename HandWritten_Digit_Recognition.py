import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# we scale the data. meaning we take every sample from 0 to 255 and scale it to
#    a value between 0 and 1. we're basically converting the samples to percentages
#    we will not scale the y data because y data is 0 1...9 they are just the digits
#    or the classifications (labels) we use to classify the samples.
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# we add a new hidden layer. the first one with 128 neurons and activation function
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))

# softmax activation function is actually determent the classification.
#   it's actually our last layer that is the output layer with 10 neurons.
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# epochs means how many times the model will see the same data over and over again
model.fit(x_train, y_train, epochs=3)


loss, accuracy = model.evaluate(x_test, y_test)

print(f'Loss is: {loss}')
print(f'Accuracy is: {accuracy}')

model.save('digits.keras')

# we recognize the digits on the image using the model we built
for digit in range(1, 6):
    img = cv.imread(f'.//digits//{digit}.png')[:, :, 0]
    img = np.invert(np.array(([img])))
    prediction = model.predict(img)
    print(f'The result is probably: {np.argmax(prediction)}') # argmax giving us the classification of the highest value
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

