# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.datasets import imdb

class IMDB(object):
	def __init__(self, batch_size, seq_len, max_vocab_size, index_from):
		self.batch_size = batch_size
		self.seq_len = seq_len
		self.max_vocab_size = max_vocab_size
		self.index_from = index_from

		(self.trainX, self.trainY), (self.testX, self.testY) = imdb.load_data(num_words=self.max_vocab_size,
			index_from=self.index_from,start_char=1, oov_char=2 ,maxlen=None)
		self.train_num_batches = len(self.trainX) // self.batch_size
		self.test_num_batches = len(self.testX) // self.batch_size
		self.vocab_size = max([max(ins) for ins in self.trainX]) + 1

		self.testX = self.fit_in_vocabulary(self.testX, self.vocab_size)
		self.trainX = self.padding(self.trainX, self.seq_len)
		self.testX = self.padding(self.testX, self.seq_len)

		word_to_index = imdb.get_word_index()
		word_to_index = {word: (index + self.index_from) for (word, index) in word_to_index.items()}
		word_to_index["<pad>"] = 0
		word_to_index["<sos>"] = 1
		word_to_index["<unk>"] = 2
		self.word_to_index = word_to_index
		self.index_to_word = {value: key for key, value in word_to_index.items()}


	def padding(self, X, seq_len):
		ret = [x[:seq_len-1] + [0] * max(seq_len-len(x), 1) for x in X]
		return np.array(ret)

	def fit_in_vocabulary(self, X, voc_size):
		return [[w for w in x if w < voc_size] for x in X]


def data_generator(X, Y, batch_size):
	size = X.shape[0]
	X_copy = X.copy()
	y_copy = Y.copy()
	indices = np.arange(size)
	np.random.shuffle(indices)
	X_copy = X_copy[indices]
	y_copy = y_copy[indices]
	i = 0
	while True:
		if i + batch_size <= size:
			yield X_copy[i : i + batch_size], y_copy[i : i + batch_size]
			i += batch_size
		else:
			i = 0
			indices = np.arange(size)
			np.random.shuffle(indices)
			X_copy = X_copy[indices]
			y_copy = y_copy[indices]
			continue