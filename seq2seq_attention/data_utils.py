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


def load_dataset(filename):

	dataset_path = os.path.join(filename)
	print("Loading dataset from {}".format(dataset_path))

	with open(dataset_path, 'rb') as f:
		data = pickle.load(f)

		word2id = data['word2id']
		id2word = data['id2word']
		trainingSamples = data['trainingSamples']

	return word2id, id2word, trainingSamples


def create_batch(samples):

	batch = Batch()

	batch.encoder_lengths = [len(sample[0]) for sample in samples]
	batch.decoder_lengths = [len(sample[1]) for sample in samples]

	max_encoder_length = max(batch.encoder_lengths) 
	max_decoder_length = max(batch.decoder_lengths)

	for sample in samples:

		# reverse encoder input and pad in front of it.
		source = list(reversed(sample[0][:max_encoder_length])) 
		padding = [PAD_ID] * (max_encoder_length - len(source))
		batch.encoder_inputs.append(padding + source)

		# add GO at the beginning and pad at the end of decoder input.
		target = sample[1][:max_decoder_length]
		padding = [PAD_ID] * (max_decoder_length - len(target))
		batch.decoder_targets.append(target + padding)
		batch.decoder_inputs.append([GO_ID] + batch.decoder_targets[-1][:-1])
	
	return batch


def get_batches(data, batch_size):

	random.shuffle(data)
	batches = []
	data_len = len(data)

	def genNextSamples():
		for i in range(0, data_len, batch_size):
			yield data[i:min(i + batch_size, data_len)]

	for samples in genNextSamples():
		batch = create_batch(samples)
		batches.append(batch)

	# return a list of batches
	return batches



def sentence_preprocess(sentence, word2id):
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

	batch = create_batch(samples)
	return batch