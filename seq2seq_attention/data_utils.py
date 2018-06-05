# -*- coding:utf-8 -*-

""" Utilities for data operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import nltk
import numpy as np
import pickle
import random

PAD_ID	= 0
GO_ID	= 1
EOS_ID	= 2
UNK_ID	= 3


class Batch(object):
	def __init__(self):
		self.encoder_inputs = []
		self.encoder_lengths = []
		self.decoder_inputs = []
		self.decoder_lengths = []
		self.decoder_targets = []

		"""
		self.encoder_inputs = []
		self.encoder_inputs_length = []
		self.decoder_targets = []
		self.decoder_targets_length = []
		"""


def loadDataset(filename):
	dataset_path = os.path.join(filename)
	print("Loading dataset from {}".format(dataset_path))
	with open(dataset_path, 'rb') as handle:
		data = pickle.load(handle)
		word2id = data['word2id']
		id2word = data['id2word']
		trainingSamples = data['trainingSamples']
	return word2id, id2word, trainingSamples


def createBatch(samples):
	batch = Batch()

	batch.encoder_lengths = [len(sample[0]) for sample in samples]
	batch.decoder_lengths = [len(sample[1]) for sample in samples]

	max_encoder_length = max(batch.encoder_lengths) 
	max_decoder_length = max(batch.decoder_lengths)

	for sample in samples:
		source = list(reversed(sample[0][:max_encoder_length]))
		padding = [PAD_ID] * (max_encoder_length - len(source))
		batch.encoder_inputs.append(padding + source)

		target = sample[1][:max_decoder_length]
		padding = [PAD_ID] * (max_decoder_length - len(target))
		batch.decoder_targets.append(target + padding)
		batch.decoder_inputs.append([GO_ID] + batch.decoder_targets[-1][:-1])
	
	return batch


def getBatches(data, batch_size):
	random.shuffle(data)
	batches = []
	data_len = len(data)
	def genNextSamples():
		for i in range(0, data_len, batch_size):
			yield data[i:min(i + batch_size, data_len)]
	for samples in genNextSamples():
		batch = createBatch(samples)
		batches.append(batch)
	return batches


def sentence2enco(sentence, word2id):
	if sentence == "":
		return None

	tokens = nltk.word_tokenize(sentence)
	if len(tokens) > 20:
		return None

	wordIds = []
	for token in tokens:
		wordIds.append(word2id.get(token, UNK_ID))

	samples =[]
	sample = (wordIds, [])
	samples.append(sample)

	batch = createBatch(samples)
	return batch