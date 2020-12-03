import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import numpy as np

from src import tfi

(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy'])

# Change to True if you want to train from scratch
train = False

if(train):
	# Save the untrained weights for future training with modified dataset
	model.save_weights('h5/fcn-untrained.h5')

	model.fit(train_images, train_labels, batch_size=100, epochs=10,
		validation_data=(test_images, test_labels))

	model.save_weights('h5/fcn-trained.h5')

else:
	model.load_weights('h5/fcn-trained.h5')

	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	print("Accuracy before faults:", test_acc)

	tfi.inject(model=model, confFile="confFiles/sample.yaml")

	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
	print("Accuracy after faults:", test_acc)