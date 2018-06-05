# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf

from data_utils import loadDataset, getBatches, sentence2enco
from model import Seq2SeqModel
from trainer import Config
from visualizer import Visualizer


def main(_):
	data_path = 'data/new-dataset-cornell-length10-filter1-vocabSize40000.pkl'
	word2id, id2word, trainingSamples = loadDataset(data_path)
	hparam = Config()
	hparam.is_training=False

	with tf.Session() as sess:
		model = Seq2SeqModel(hparam, word2id)
		ckpt = tf.train.get_checkpoint_state(hparam.save_path)
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			print("Restoring model parameters from %s." % ckpt.model_checkpoint_path)
			model.saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			print("Creating model with fresh parameters.")
			sess.run(model.init)
		
		sys.stdout.write("> ")
		sys.stdout.flush()
		sentence = sys.stdin.readline()
		while sentence:
			batch = sentence2enco(sentence, word2id)
			outputs = model.infer_session(sess, batch)

			predicted_ids = outputs["predicted_ids"]
			out_sents = [id2word[idx] for idx in predicted_ids[0][0].tolist()]
			print(" ".join(out_sents))
			print("> ", "")
			sys.stdout.flush()
			sentence = sys.stdin.readline()


if __name__ == '__main__':
	tf.app.run()