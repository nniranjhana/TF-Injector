#!/usr/bin/python

import tensorflow as tf
from struct import pack, unpack

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import numpy as np
from tensorflow.keras import Model, layers, datasets
import random, math
from src import config

def bitflip(f, pos):
    f_ = pack('f', f)
    b = list(unpack('BBBB', f_))
    [q, r] = divmod(pos, 8)
    b[q] ^= 1 << r
    f_ = pack('BBBB', *b)
    f = unpack('f', f_)
    return f[0]

class inject_model():
	def __init__(self, model, confFile="confFiles/sample.yaml"):
		fiConf = config.mconfig(confFile)
		self.Model = model # No more passing or using a session variable in TF v2
		fiFunc = getattr(self, fiConf["Type"])
		fiFunc(model, fiConf)

	def shuffle(self, model, fiConf):
		v = model.trainable_variables[fiConf["Artifact"]] # No tf.get_collection in TF v2
		v_ = tf.random.shuffle(v) # No tf.random_shuffle in TF v2
		v.assign(v_) # No tf.assign in TF v2

	def zeros(self, model, fiConf):
		v = model.trainable_variables[fiConf["Artifact"]]
		num = v.shape.num_elements()
		sz = (fiConf["Amount"] * num) / 100
		sz = math.floor(sz) # Python 2.7 returns int, but need explicit rounding for Python 3.5
		ind = random.sample(range(num), sz)
		i, j = v.get_shape().as_list()
		ind_ = []
		for item in ind:
			ind_.append([item/j, item%j])
		upd = tf.zeros([sz], tf.float32)
		ind_ = tf.cast(ind_, tf.int32) # TF v1 returns int, but need explicit cast for TF v2
		print(ind_)
		v_ = tf.tensor_scatter_nd_update(v, ind_, upd) # Need tf.tensor_ for TF v2
		v.assign(v_)

	def mutate(self, model, fiConf):
		v = model.trainable_variables[fiConf["Artifact"]]
		num = v.shape.num_elements()
		sz = fiConf["Amount"]
		ind = random.sample(range(num), sz)
		i, j = v.get_shape().as_list()
		ind_ = []
		for item in ind:
			ind_.append([item/j, item%j])
		ind_ = tf.cast(ind_, tf.int32)
		upd = []
		if (fiConf["Bit"]=='N'):
			for item in ind_:
				val = v[item[0]][item[1]]
				pos = random.randint(0, 31)
				val_ = bitflip(val, pos)
				upd.append(val_)
		else:
			pos = fiConf["Bit"]
			for item in ind_:
				val = v[item[0]][item[1]]
				val_ = bitflip(val, pos)
				upd.append(val_)
		v_ = tf.tensor_scatter_nd_update(v, ind_, upd)
		v.assign(v_)

	def metamorph(self, model, fiConf):
		(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
		train_images, test_images = train_images / 255.0, test_images / 255.0

		permute = fiConf["Mutation"]

		color = {
			'R' : 0,
			'G' : 1,
			'B' : 2
		}

		# Build up the training and testing dataset according to the specified color permutation
		(train_images_, test_images_) = (train_images[:,:,:,color[permute[0]]:(color[permute[0]]+1)],
			test_images[:,:,:,color[permute[0]]:(color[permute[0]]+1)])
		(train_images_, test_images_) = (np.concatenate((train_images_, train_images[:,:,:,color[permute[1]]:(color[permute[1]]+1)]), axis = 3),
			np.concatenate((test_images_, test_images[:,:,:,color[permute[1]]:(color[permute[1]]+1)]), axis = 3))
		(train_images_, test_images_) = (np.concatenate((train_images_, train_images[:,:,:,color[permute[2]]:(color[permute[2]]+1)]), axis = 3),
			np.concatenate((test_images_, test_images[:,:,:,color[permute[2]]:(color[permute[2]]+1)]), axis = 3))

		# Re-train the model with the mutated color dataset
		model.fit(train_images_, train_labels, epochs=10,
			validation_data=(test_images_, test_labels))

		test_loss, test_acc = model.evaluate(test_images_,  test_labels, verbose=2)
		print("Accuracy with the permutated", permute, "dataset:", test_acc)

def inject_data(x_test, confFile="confFiles/sample.yaml"):
	fiConf = config.dconfig(confFile)
	fiFunc = globals()[fiConf["Type"]]
	return fiFunc(x_test, fiConf)

def shuffle(x_test, fiConf):
	x_test_ = tf.random.shuffle(x_test)
	return x_test_

def repeat(x_test, fiConf):
	num = x_test.shape[0]
	rep_sz = fiConf["Amount"]
	rep_sz = (rep_sz * num) / 100
	rep_sz = math.floor(rep_sz)
	sz = num - rep_sz
	ind = random.sample(range(num), sz)
	x_test_ = tf.gather(x_test, ind)
	upd = random.sample(ind, rep_sz)
	x_ = tf.gather(x_test, upd)
	x_test_ = tf.concat([x_test_, x_], 0)
	return x_test_

'''
Outdated TF v1 test code, kept in case we need to support tests in TF v1 in the future

	def __init__(self, sess): #config = "conf/default.yaml"):

		self.session = sess
		v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0]
		print (sess.run(v[0][0]))
		v_ = tf.scatter_nd_update(v, tf.constant([[0, 0], [0, 1], [0, 2]]), tf.constant([100., 20000., 300000.]))
		sess.run(tf.assign(v, v_))
		print(sess.run(v[0][0]))
'''