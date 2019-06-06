# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
from keras.datasets import imdb
from data_utils import IMDB, data_generator
from model import RNNTextClassification

import numpy as np
import tensorflow as tf

class Config:
    vocab_size = -1
    max_vocab_size = 10000
    index_from = 3
    seq_len = 250
    embedding_dim = 128
    hidden_dim = 200
    attention_dim = 64
    keep_prob = 0.8
    batch_size = 64
    num_epoch = 1000
    num_layers = 1
    clip_norm = 5.0
    save_path = 'save'


def main(_):
	hparam = Config()
	dataloader = IMDB(hparam.batch_size, hparam.seq_len, hparam.max_vocab_size, hparam.index_from)
	hparam.vocab_size = dataloader.vocab_size
	model = RNNTextClassification(hparam)

	with tf.Session() as sess:
		sess.run(model.init)
		train_writer = tf.summary.FileWriter(hparam.save_path+'/train', sess.graph)
		test_writer = tf.summary.FileWriter(hparam.save_path+'/test', sess.graph)
		
		train_generator = data_generator(dataloader.trainX, dataloader.trainY, hparam.batch_size)
		test_generator = data_generator(dataloader.testX, dataloader.testY, hparam.batch_size)

		print("Start Learning.")
		best_test_results = 10000
		for epoch in range(hparam.num_epoch):

			print(" Starting Epoch [%d/%d]:" % (epoch, hparam.num_epoch))
			train_loss = 0.0
			train_accuracy = 0.0
			train_num_batches = dataloader.train_num_batches

			for b in tqdm(range(train_num_batches)):

				x_batch, y_batch = next(train_generator)
				seq_len = np.array([list(x).index(0) + 1 for x in x_batch])
				lo, acc, _, summ = sess.run([model.loss, model.accuracy, model.train_op, model.merged],
					feed_dict={model.inputs:x_batch, 
								model.targets: y_batch, 
								model.lengths:seq_len, 
								model.keep_prob:hparam.keep_prob})
				train_loss += lo
				train_accuracy += acc
				train_writer.add_summary(summ, b + train_num_batches * epoch)

			train_loss /= train_num_batches
			train_accuracy /= train_num_batches

			test_loss = 0.0
			test_accuracy = 0.0
			test_num_batches = dataloader.test_num_batches

			for b in tqdm(range(test_num_batches)):
				test_x_batch, test_y_batch = next(test_generator)
				test_seq_len = np.array([list(x).index(0) + 1 for x in test_x_batch])

				lo_test, acc_test, summ_test = sess.run([model.loss, model.accuracy, model.merged],
					feed_dict={model.inputs:test_x_batch, 
					model.targets:test_y_batch, 
					model.lengths:test_seq_len, 
					model.keep_prob:1.0})

				test_loss += lo_test
				test_accuracy += acc_test
				test_writer.add_summary(summ_test, b + test_num_batches * epoch)

			test_loss /= test_num_batches
			test_accuracy /= test_num_batches

			print(" loss: {:.6f} | test_loss: {:.6f} | acc: {:.6f} | test_acc: {:.6f}".format(
				train_loss, test_loss, train_accuracy, test_accuracy))

			train_writer.close()
			test_writer.close()

			if best_test_results > test_loss:
				best_test_results = test_loss
				model.saver.save(sess, hparam.save_path + "/model.ckpt")
				print("Run 'tensorboard --logdir=save' to checkout tensorboard logs.")


if __name__ == '__main__':
	tf.app.run()