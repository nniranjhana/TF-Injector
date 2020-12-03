import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import numpy as np
import sys

from src import tfi

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy'])

try:
	check = sys.argv[1]
	assert check == "train" or "test"
except:
	print("Provide either the 'train' or 'test' argument to run.")
	sys.exit()

if(check == "train"):
	# Save the untrained weights for future training with modified dataset
	model.save_weights('h5/nn-untrained.h5')

	model.fit(train_images, train_labels, epochs=5,
		validation_data=(test_images, test_labels))

	model.save_weights('h5/nn-trained.h5')

else:
	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	print("Accuracy before faults:", test_acc)

	model.load_weights('h5/nn-trained.h5')

	tfi.inject(model=model, confFile="confFiles/sample.yaml")

	test_loss, test_acc = model.evaluate(test_images,  test_labels)
	print("Accuracy after faults:", test_acc)