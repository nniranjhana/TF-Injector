#!/usr/bin/python
import re

import tensorflow as tf
from struct import pack, unpack

import numpy as np
from tensorflow.keras import Model, layers, datasets
import random, math
from src import config


def inject(confFile="confFiles/sample.yaml", **kwargs):
	if kwargs.get('fiConf'):
		fiConf = kwargs.pop('fiConf')
	else:
		fiConf = config.config(confFile)
	fiFunc = globals()[fiConf["Type"]]
	return fiFunc(fiConf, **kwargs)

def shuffle_params(fiConf, **kwargs):
	model = kwargs["model"]
	v = model.trainable_variables[fiConf["Artifact"]] # No tf.get_collection in TF v2
	v_ = tf.random.shuffle(v) # No tf.random_shuffle in TF v2
	v.assign(v_) # No tf.assign in TF v2

def zeros(fiConf, **kwargs):
	model = kwargs["model"]
	v = model.trainable_variables[fiConf["Artifact"]]
	num = v.shape.num_elements()
	sz = (fiConf["Amount"] * num) / 100
	sz = math.floor(sz) # Python 2.7 returns int, but need explicit rounding for Python 3.5
	ind = random.sample(range(num), sz)
	elem_shape = v.shape
	v_ = tf.identity(v)
	v_ = tf.keras.backend.flatten(v_)
	v_ = tf.unstack(v_)
	for item in ind:
		v_[item] = 0.
	v_ = tf.stack(v_)
	v_ = tf.reshape(v_, elem_shape)
	v.assign(v_)

def bitflip(f, pos):
    f_ = pack('f', f)
    b = list(unpack('BBBB', f_))
    [q, r] = divmod(pos, 8)
    b[q] ^= 1 << r
    f_ = pack('BBBB', *b)
    f = unpack('f', f_)
    return f[0]

def mutate(fiConf, **kwargs):
	model = kwargs["model"]
	v = model.trainable_variables[fiConf["Artifact"]]
	num = v.shape.num_elements()
	sz = fiConf["Amount"]
	ind = random.sample(range(num), sz)
	elem_shape = v.shape
	v_ = tf.identity(v)
	v_ = tf.keras.backend.flatten(v_)
	v_ = tf.unstack(v_)
	regex = re.compile('(\d+)-(\d+)')

	for item in ind:
		val = v_[item]
		if(fiConf["Bit"]=='N'):
			pos = random.randint(0, 31)
		else:
			pos = random.randint(*map(int, regex.match(fiConf['Bit']).groups()))
		val_ = bitflip(val, pos)
		v_[item] = val_
	v_ = tf.stack(v_)
	v_ = tf.reshape(v_, elem_shape)
	v.assign(v_)

def shuffle(fiConf, **kwargs):
	x_test = kwargs["x_test"]
	y_test = kwargs["y_test"]
	num = x_test.shape[0]
	ind = tf.range(0, num)
	ind_ = tf.random.shuffle(ind)
	x_test_, y_test_ = tf.gather(x_test, ind_), tf.gather(y_test, ind_)
	return (x_test_, y_test_)

def repeat(fiConf, **kwargs):
	x_test = kwargs["x_test"]
	y_test = kwargs["y_test"]
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

def remove(fiConf, **kwargs):
	x_test = kwargs["x_test"]
	y_test = kwargs["y_test"]
	num = x_test.shape[0]
	rem_sz = fiConf["Amount"]
	rem_sz = (rem_sz * num) / 100
	rem_sz = math.floor(rem_sz)
	sz = num - rem_sz
	ind = random.sample(range(num), sz)
	x_test_, y_test_ = tf.gather(x_test, ind), tf.gather(y_test, ind)
	return (x_test_, y_test_)

def noise_add(fiConf, **kwargs):
	x_test = kwargs["x_test"]
	num = x_test.size # Total elements from all datapoints
	sz = len(x_test) # Number of datapoints
	elem_shape = x_test.shape[1:] # Extract each element's shape as a tuple for reshape later
	add_sz = num//sz # Number of elements in each datapoint
	err_sz = fiConf["Amount"]
	err_sz = (err_sz * sz) / 100 # Number of datapoints to add noise to
	err_sz = math.floor(err_sz)
	ind = random.sample(range(sz), err_sz)

	if(fiConf["Mutation"] == "Random"):
		for item in ind:
			upd = np.random.standard_normal(add_sz)
			x_test_ = x_test[item].flatten()
			x_test_ = x_test_ + upd
			x_test[item] = x_test_.reshape(elem_shape)

	elif(fiConf["Mutation"] == "Gauss"):
		try:
			r, c, ch = x_test[0].shape
		except:
			r, c = x_test[0].shape
			ch = 1
		m = 0; v = 0.1
		s = v**0.5
		gauss = np.random.normal(m, s, (r, c, ch))
		gauss = gauss.reshape(r, c, ch)
		for item in ind:
			try:
				x_test[item] = x_test[item] + gauss
			except:
				gauss = gauss.reshape(r,c)
				x_test[item] = x_test[item] + gauss

	elif(fiConf["Mutation"] == "SP"):
		try:
			r, c, ch = x_test[0].shape
		except:
			r, c = x_test[0].shape
			ch = 1
		sp = 0.5; a = 0.04
		for item in ind:
			salt = np.ceil(a*(x_test[item].size)*sp)
			co = [np.random.randint(0, i-1, int(salt))
				for i in x_test[item].shape]
			x_test[item][co] = 1
			pepper = np.ceil(a*(x_test[item].size)*(1.-sp))
			co = [np.random.randint(0, i-1, int(pepper))
				for i in x_test[item].shape]
			x_test[item][co] = 0

	elif(fiConf["Mutation"] == "Speckle"):
		try:
			r, c, ch = x_test[0].shape
		except:
			r, c = x_test[0].shape
			ch = 1
		speckle = np.random.randn(r, c, ch)
		if(ch == 1):
			speckle = speckle.reshape(r, c)
		else:
			speckle = speckle.reshape(r, c, ch)
		for item in ind:
			x_test[item] = x_test[item] + x_test[item]*speckle*0.5

	return x_test

def label_err(fiConf, **kwargs):
	y_test = kwargs["y_test"]
	num = y_test.shape[0]
	err_sz = fiConf["Amount"]
	err_sz = (err_sz * num) / 100
	err_sz = math.floor(err_sz)
	ind = random.sample(range(num), err_sz)
	_, check = str(y_test.shape).split(",")
	if(check==')'):
		y_test = y_test.reshape(num, 1)
	for item in ind:
		r = list(range(0, y_test[item][0])) + list(range(y_test[item][0] + 1, 10))
		y_test[item] = random.choice(r)
	return y_test

def metamorph_color(fiConf, **kwargs):
	'''
	MR applicability: Permutation of input channels applies only to certain RGB datasets
	'''

	x_test = kwargs["x_test"]
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

def metamorph_constant(fiConf, **kwargs):
	'''
	MR applicability: Shift of train and test features by a constant applies only for RBF kernel
	'''

	x_test = kwargs["x_test"]
	b = float(fiConf["Mutation"])
	x_test_ = x_test + b
	return x_test_

def metamorph_linear(fiConf, **kwargs):
	'''
	MR applicability: Linear scaling of test features applies only for linear kernel
	'''

	x_test = kwargs["x_test"]
	linear = float(fiConf["Mutation"])
	W, b = linear.replace(' ', '').split(',')
	W, b = float(W), float(b)
	x_test_ = x_test*W + b
	return x_test_

def class_add(fiConf, **kwargs):
	x_name = kwargs["x_name"]
	x_test = kwargs["x_test"]
	y_test = kwargs["y_test"]
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