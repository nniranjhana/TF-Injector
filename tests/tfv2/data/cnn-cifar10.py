import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import time

from src import softtensorfi

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Save the untrained weights for future training with modified dataset
model.save_weights('linear0.h5')

model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Save the model and trained weights
model.save('RGB')
model.save_weights('RGB.h5')

# Load the model and trained weights
model = tf.keras.models.load_model('RGB')
model.load_weights('RGB.h5')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Accuracy with the original dataset:", test_acc)

# Load the untrained weights for retraining with permutated dataset
model.load_weights('RGB0.h5')

train_images_ = softtensorfi.inject_data(train_images)
test_images_ = softtensorfi.inject_data(test_images)

test_loss, test_acc = model.evaluate(test_images_,  test_labels, verbose=2)
print("Accuracy with the modified dataset before training:", test_acc)

model.fit(train_images_, train_labels, epochs=10, 
                    validation_data=(test_images_, test_labels))

test_loss, test_acc = model.evaluate(test_images_,  test_labels, verbose=2)
print("Accuracy with the modified dataset after training:", test_acc)