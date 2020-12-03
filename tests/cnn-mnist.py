import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import numpy as np

from src import tfi

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Change to True if you want to train from scratch
train = False

if(train):
	# Save the untrained weights for future training with modified dataset
	model.save_weights('h5/cnnm-untrained.h5')

	model.fit(train_images, train_labels, batch_size=100, epochs=10,
		validation_data=(test_images, test_labels))

	model.save_weights('h5/cnnm-trained.h5')

else:
	model.load_weights('h5/cnnm-trained.h5')

	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	print("Accuracy before faults:", test_acc)

	tfi.inject(model=model, confFile="confFiles/sample.yaml")

	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
	print("Accuracy after faults:", test_acc)