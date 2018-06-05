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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Visualizer(object):
	def __init__(self, alignment, input_list, output_list):
		activation_map = np.squeeze(alignment) # [dec_seq_len, enc_seq_len]

		plt.clf()
		f = plt.figure(figsize=(8, 8.5))
		ax = f.add_subplot(1, 1, 1)

		# add image
		i = ax.imshow(activation_map, interpolation='nearest', cmap='gray')
		
		# add colorbar
		cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
		cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
		cbar.ax.set_xlabel('Probability', labelpad=2)

		### config Y
		# ax.yaxis.set_ticks_position('left')
		ax.set_yticks(range(len(output_list)))
		ax.set_yticklabels(output_list)
		ax.set_ylabel('Output Sequence')
		
		### config X
		# ax.xaxis.set_ticks_position('top')
		ax.set_xticks(range(len(input_list)))
		ax.set_xticklabels(input_list)
		ax.set_xlabel('Input Sequence')
		

		ax.grid()
		# ax.legend(loc='best')
		f.savefig("attention_maps.jpg", bbox_inches='tight')


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
			in_sents = [id2word[idx] for idx in batch.encoder_inputs[0]]

			print(in_sents)
			print(out_sents)
			Visualizer(outputs["alignments"], in_sents, out_sents)


			print(" ".join(out_sents))
			print("> ", "")
			sys.stdout.flush()
			sentence = sys.stdin.readline()


if __name__ == '__main__':
	tf.app.run()