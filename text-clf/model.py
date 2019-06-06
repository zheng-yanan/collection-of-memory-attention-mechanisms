# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class RNNTextClassification(object):

	def __init__(self, hparam):

		self.seq_len = hparam.seq_len
		self.vocab_size = hparam.vocab_size
		self.embedding_dim = hparam.embedding_dim
		self.num_layers = hparam.num_layers
		self.hidden_dim = hparam.hidden_dim
		self.attention_dim = hparam.attention_dim
		self.clip_norm = hparam.clip_norm

		self.inputs = tf.placeholder(tf.int32, [None, self.seq_len])
		self.targets = tf.placeholder(tf.float32, [None])
		self.lengths = tf.placeholder(tf.int32, [None])
		self.keep_prob = tf.placeholder(tf.float32, [])

		self.global_step = tf.Variable(0, trainable=False)
		self.lr = tf.Variable(1e-3, trainable=False)

		embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0), trainable=True)
		embedded_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

		def create_rnn_unit(reuse):
			cell = tf.contrib.rnn.GRUCell(self.hidden_dim, reuse=reuse)
			# cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
			return cell

		fw_cell = tf.nn.rnn_cell.MultiRNNCell([create_rnn_unit(reuse=None) for _ in range(self.num_layers)],  state_is_tuple=True)
		bw_cell = tf.nn.rnn_cell.MultiRNNCell([create_rnn_unit(reuse=None) for _ in range(self.num_layers)],  state_is_tuple=True)
		rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
			inputs=embedded_inputs, sequence_length=self.lengths, dtype=tf.float32)

		attention_output, self.alphas = self.attention(rnn_outputs)
		drop_attention = tf.nn.dropout(attention_output, self.keep_prob)

		W = tf.Variable(tf.truncated_normal([2 * self.hidden_dim, 1], stddev=0.1))
		b = tf.Variable(tf.constant(0., shape=[1]))
		y_hat = tf.nn.xw_plus_b(drop_attention, W, b)
		y_hat = tf.squeeze(y_hat)

		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=self.targets))
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), self.targets), tf.float32))
		
		tf.summary.scalar("loss", self.loss)
		tf.summary.scalar("accuracy", self.accuracy)

		params = tf.trainable_variables()
		gradients = tf.gradients(self.loss, params)
		clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
		self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

		self.saver = tf.train.Saver()
		self.merged = tf.summary.merge_all()
		self.init = tf.global_variables_initializer()


	def get_parameter_size(self):
		all_vars = tf.global_variables()
		total_count = 0
		for item in all_vars:
			if "Adam" in item.name:
				continue
			shape = item.get_shape().as_list()
			if len(shape) == 0:
				total_count += 1
			else:
				size =  1
				for val in shape:
					size *= val
				total_count += size
		return total_count


	### MLP attention
	def attention(self, rnn_outputs):
			if isinstance(rnn_outputs, tuple):
				inputs = tf.concat(rnn_outputs, 2)
				hidden_dim = self.hidden_dim * 2
			else:
				inputs = rnn_outputs
				hidden_dim = self.hidden_dim
			
			w_omega = tf.Variable(tf.random_normal([hidden_dim, self.attention_dim], stddev=0.1))
			b_omega = tf.Variable(tf.random_normal([self.attention_dim], stddev=0.1))
			u_omega = tf.Variable(tf.random_normal([self.attention_dim], stddev=0.1))

			new_inputs = tf.reshape(inputs, [-1, hidden_dim])
			v = tf.tanh(tf.matmul(new_inputs, w_omega) + b_omega)

			vu = tf.reduce_sum(tf.multiply(v, u_omega), -1)
			vu = tf.reshape(vu, [-1, self.seq_len])

			alphas = tf.nn.softmax(vu, name='alphas')
			outputs = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

			return outputs, alphas