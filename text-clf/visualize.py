# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
from keras.datasets import imdb
from data_uitls import IMDB, data_generator
from model import RNNTextClassification

import numpy as np
import tensorflow as tf
from trainer import Config


def main(_):
	hparam = Config()
	dataloader = IMDB(hparam.batch_size, hparam.seq_len, hparam.max_vocab_size, hparam.index_from)
	hparam.vocab_size = dataloader.vocab_size
	model = RNNTextClassification(hparam)

	with tf.Session() as sess:

		model.saver.restore(sess, hparam.save_path+"/model.ckpt")
		x_batch_test, y_batch_test = dataloader.testX[:5], dataloader.testY[:5]
		seq_len_test = np.array([list(x).index(0) + 1 for x in x_batch_test])

		alphas_test = sess.run(model.alphas, 
			feed_dict={
			model.inputs: x_batch_test, 
			model.targets: y_batch_test, 
			model.lengths: seq_len_test, 
			model.keep_prob: 1.0})

		word_list = []
		for i in range(5):
			words = list(map(dataloader.index_to_word.get, x_batch_test[i]))
			word_list.append(words)
		
		# Save visualization as HTML
		with open("visualization.html", "w") as html_file:
			for i in range(5):
				words = word_list[i]
				alphas_values = alphas_test[i]
				for word, alpha in zip(words, alphas_values / alphas_values.max()):
					if word == "<sos>":
						continue
					elif word == "<pad>":
						break
					html_file.write('<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (alpha, word))
				html_file.write("</br></br>")

if __name__ == '__main__':
	tf.app.run()		