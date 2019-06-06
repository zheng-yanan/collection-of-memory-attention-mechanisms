# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
from tqdm import tqdm
import tensorflow as tf
from data_utils import load_dataset, get_batches, sentence_preprocess
from model import Seq2SeqModel


tf.app.flags.DEFINE_boolean('is_training', True, '')
tf.app.flags.DEFINE_boolean('resume', False, '')
FLAGS = tf.app.flags.FLAGS


class Config:

	learning_rate = 0.001
	embedding_dim = 200
	num_layers = 2
	hidden_dim = 200
	max_gradient_norm = 5.0
	is_training = FLAGS.is_training
	rnn_type = "lstm"
	attention_option = "luong"

	batch_size = 128
	num_epoch = 1000
	display_per_step = 100
	save_path = "save"
	model_name = "attention_seq2seq.ckpt"


def main(_):

	data_path = 'data/new-dataset-cornell-length10-filter1-vocabSize40000.pkl'
	word2id, id2word, trainingSamples = load_dataset(data_path)
	hparam = Config()

	with tf.Session() as sess:

		model = Seq2SeqModel(hparam, word2id)
		ckpt = tf.train.get_checkpoint_state(hparam.save_path)

		if FLAGS.resume and ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			print("Restoring model parameters from %s." % ckpt.model_checkpoint_path)
			model.saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			print("Creating model with fresh parameters.")
			sess.run(model.init)

		train_writer = tf.summary.FileWriter(hparam.save_path, graph=sess.graph)

		for epoch in range(hparam.num_epoch):
			print("Starting Epoch {}/{}:".format(epoch, hparam.num_epoch))
			
			batches = get_batches(trainingSamples, hparam.batch_size)
			total_loss = 0.0
			total_count = 0

			for nextBatch in tqdm(batches, desc="training"):

				outputs = model.train_session(sess, nextBatch)

				loss = outputs["loss"]
				summary = outputs["summary"]
				step = outputs["step"]
				train_writer.add_summary(summary, step)
				total_loss += loss
				total_count += 1

				if step % hparam.display_per_step == 0:

					perplexity = math.exp(float(total_loss / total_count)) if total_loss / total_count < 300 else float('inf')
					tqdm.write(" Step %d | Per-word Loss %.4f | Perplexity %.4f" % (step, total_loss / total_count, perplexity))

					checkpoint_path = os.path.join(hparam.save_path, hparam.model_name)
					model.saver.save(sess, checkpoint_path)


			tqdm.write("\n")
			tqdm.write(" Epoch %d | Per-word Loss %.4f | Perplexity %.4f" % (epoch, total_loss / total_count, perplexity))
			tqdm.write("\n")



if __name__ == '__main__':
	tf.app.run()