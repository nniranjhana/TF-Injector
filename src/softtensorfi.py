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

def inject_data(x_test, y_test, confFile="confFiles/sample.yaml"):
	fiConf = config.dconfig(confFile)
	fiFunc = globals()[fiConf["Type"]]
	return fiFunc(x_test, y_test, fiConf)

def shuffle(x_test, y_test, fiConf):
	num = x_test.shape[0]
	ind = tf.range(0, num)
	ind_ = tf.random.shuffle(ind)
	x_test_, y_test_ = tf.gather(x_test, ind_), tf.gather(y_test, ind_)
	return (x_test_, y_test_)

def repeat(x_test, y_test, fiConf):
	num = x_test.shape[0]
	rep_sz = fiConf["Amount"]
	rep_sz = (rep_sz * num) / 100
	rep_sz = math.floor(rep_sz)
	sz = num - rep_sz
	ind = random.sample(range(num), sz)
	x_test_, y_test_ = tf.gather(x_test, ind), tf.gather(y_test, ind)
	upd = random.sample(ind, rep_sz)
	x_, y_ = tf.gather(x_test, upd), tf.gather(y_test, upd)
	x_test_, y_test_ = tf.concat([x_test_, x_], 0), tf.concat([y_test_, y_], 0)
	return (x_test_, y_test_)

def remove(x_test, y_test, fiConf):
	num = x_test.shape[0]
	rem_sz = fiConf["Amount"]
	rem_sz = (rem_sz * num) / 100
	rem_sz = math.floor(rem_sz)
	sz = num - rem_sz
	ind = random.sample(range(num), sz)
	x_test_, y_test_ = tf.gather(x_test, ind), tf.gather(y_test, ind)
	return (x_test_, y_test_)

def inject_data(xy_test, confFile="confFiles/sample.yaml"):
	fiConf = config.dconfig(confFile)
	fiFunc = globals()[fiConf["Type"]]
	return fiFunc(xy_test, fiConf)

def noise_add(x_test, fiConf):
	num = x_test.size # Total elements from all datapoints
	sz = len(x_test) # Number of datapoints
	elem_shape = x_test.shape[1:] # Extract each element's shape as a tuple for reshape later
	add_sz = num//sz # Number of elements in each datapoint
	err_sz = fiConf["Amount"]
	err_sz = (err_sz * sz) / 100 # Number of datapoints to add noise to
	err_sz = math.floor(err_sz)
	ind = random.sample(range(sz), err_sz)
	for item in ind:
		upd = np.random.standard_normal(add_sz)
		x_test_ = x_test[item].flatten()
		x_test_ = x_test_ + upd
		x_test[item] = x_test_.reshape(elem_shape)
	return x_test

def label_err(y_test, fiConf):
	num = y_test.shape[0]
	err_sz = fiConf["Amount"]
	err_sz = (err_sz * num) / 100
	err_sz = math.floor(err_sz)
	ind = random.sample(range(num), err_sz)
	for item in ind:
		r = list(range(0, y_test[item][0])) + list(range(y_test[item][0] + 1, 10))
		y_test[item] = random.choice(r)
	return y_test

def inject_data(x_name, x_test, y_test, confFile="confFiles/sample.yaml"):
	fiConf = config.dconfig(confFile)
	fiFunc = globals()[fiConf["Type"]]
	return fiFunc(x_name, x_test, y_test, fiConf)	

def class_add(x_name, x_test, y_test, fiConf):
	import tensorflow_datasets as tfds
	ds = tfds.load(x_name, split='train', shuffle_files=True)
	for dp in ds.take(1):
		dl = list(dp.keys())
		elem_shape = dp["image"].shape
	if(elem_shape != x_test.shape[1:]):
		raise AssertionError("Datasets' input shapes don't match")
	add_sz = fiConf["Amount"]
	upd = ds.take(add_sz)
	x_test_, y_test_ = [], []
	for item in tfds.as_numpy(upd):
		x_test_.append(item["image"])
	x_test = np.append(x_test, x_test_, axis = 0)
	ind = random.sample(range(y_test.shape[0]), add_sz)
	for i in ind:
		y_test_.append(y_test[i])
	y_test = np.append(y_test, y_test_, axis = 0)
	return x_test, y_test

def metamorph_color(x_test, fiConf):
	'''
	MR applicability: Permutation of input channels applies only to certain RGB datasets
	'''

	permute = fiConf["Mutation"]

	color = {
		'R' : 0,
		'G' : 1,
		'B' : 2
	}

	# Build up the dataset according to the specified color permutation
	x_test_ = x_test[:,:,:,color[permute[0]]:(color[permute[0]]+1)]
	x_test_ = np.concatenate((x_test_, x_test[:,:,:,color[permute[1]]:(color[permute[1]]+1)]), axis = 3)
	x_test_ = np.concatenate((x_test_, x_test[:,:,:,color[permute[2]]:(color[permute[2]]+1)]), axis = 3)

	return x_test_

def metamorph_constant(x_test, fiConf):
	'''
	MR applicability: Shift of train and test features by a constant applies only for RBF kernel
	'''

	b = float(fiConf["Mutation"])
	x_test_ = x_test + b
	return x_test_

def metamorph_linear(x_test, fiConf):
	'''
	MR applicability: Linear scaling of test features applies only for linear kernel
	'''

	linear = float(fiConf["Mutation"])
	W, b = linear.replace(' ', '').split(',')
	W, b = float(W), float(b)
	x_test_ = x_test*W + b
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