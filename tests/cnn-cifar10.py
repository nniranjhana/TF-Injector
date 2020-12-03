import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import numpy as np

from src import tfi

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
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

# Change to True if you want to train from scratch
train = False

if(train):
    # Save the untrained weights for future training with modified dataset
    model.save_weights('h5/cnnc-untrained.h5')

    model.fit(train_images, train_labels, batch_size=100, epochs=5,
        validation_data=(test_images, test_labels))

    model.save_weights('h5/cnnc-trained.h5')

else:
    model.load_weights('h5/cnnc-trained.h5')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print("Accuracy before faults:", test_acc)

    tfi.inject(model=model, confFile="confFiles/sample.yaml")

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print("Accuracy after faults:", test_acc)