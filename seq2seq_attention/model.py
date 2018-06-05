# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import data_utils
import tensorflow as tf


class Seq2SeqModel(object):
	def __init__(self, hparam, word_to_index):
		self.learning_rate = hparam.learning_rate
		self.embedding_dim = hparam.embedding_dim
		self.num_layers = hparam.num_layers
		self.word_to_index = word_to_index
		self.vocab_size = len(word_to_index)
		self.hidden_dim = hparam.hidden_dim
		self.max_gradient_norm = hparam.max_gradient_norm
		self.is_training = hparam.is_training
		self.rnn_type = hparam.rnn_type
		self.attention_option = hparam.attention_option
		
		self._build_global_helper()
		self._build_encoder_graph()
		self._build_decoder_graph()
		self._build_infer_graph()

		if self.is_training:
			self._build_backward_graph()
			self._build_summary_graph()
			self.merged = tf.summary.merge_all()

		self.saver = tf.train.Saver()
		self.init = tf.global_variables_initializer()

	def _create_rnn_cell(self, reuse):
		def single_rnn_cell():
			if self.rnn_type == "lstm":
				single_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim, reuse=reuse)
			elif self.rnn_type == "gru":
				single_cell = tf.contrib.rnn.GRUCell(self.hidden_dim, reuse=reuse)
			else:
				raise ValueError("Unsupported rnn_type.")
			cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
			return cell
		cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
		return cell

	def _build_global_helper(self):
		self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name="encoder_inputs")
		self.encoder_lengths = tf.placeholder(tf.int32, [None], name="encoder_lengths")
		self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")
		self.decoder_targets = tf.placeholder(tf.int32, [None, None], name="decoder_targets")
		self.decoder_lengths = tf.placeholder(tf.int32, [None], name='decoder_lengths')

		self.batch_size = tf.shape(self.encoder_inputs)[0]
		self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

		self.max_decoder_length = tf.reduce_max(self.decoder_lengths, name="max_decoder_length")
		self.max_encoder_length = tf.reduce_max(self.encoder_lengths, name="max_encoder_length")
		self.mask = tf.sequence_mask(self.decoder_lengths, self.max_decoder_length, dtype=tf.float32, name='masks')

		self.lr = tf.Variable(self.learning_rate, trainable=False)
		self.new_lr = tf.placeholder(tf.float32, [])
		self.lr_update = tf.assign(self.lr, self.new_lr)

		self.global_step = tf.Variable(0, trainable=False)


	def _build_encoder_graph(self):
		with tf.variable_scope("encoder"):
			with tf.device("/cpu:0"):
				self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_dim], dtype=tf.float32)
			self.embedded_encoder_inputs = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
			encoder_cell = self._create_rnn_cell(reuse=None)
			self.encoder_outputs, self.encoder_states = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=self.embedded_encoder_inputs,
				sequence_length=self.encoder_lengths, dtype=tf.float32)

	def _build_decoder_graph(self):
		with tf.variable_scope("decoder", reuse=None):
			decoder_cell = self._create_rnn_cell(reuse=None)
			attention_mechanism = self._create_attention(self.encoder_outputs, self.encoder_lengths)
			
			attention_wrapper = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, 
				attention_mechanism=attention_mechanism, attention_layer_size=self.hidden_dim, alignment_history=True)
			attention_wrapper_init = attention_wrapper.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_states)
			
			output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), 
				_scope='decoder/dense', _reuse=None)
			embedded_decoder_inputs = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)

			training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=embedded_decoder_inputs,
				sequence_length=self.decoder_lengths, time_major=False)
			training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_wrapper, helper=training_helper,
				initial_state=attention_wrapper_init, output_layer=output_layer)

			decoder_outputs, decoder_final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder, 
				impute_finished=True, maximum_iterations=self.max_decoder_length)
			self.train_logits = tf.identity(decoder_outputs.rnn_output)
			self.train_predicted_ids = tf.argmax(self.train_logits, axis=-1, name='train_predicted_ids')
			self.train_alignments = tf.transpose(decoder_final_state.alignment_history.stack(), [1,2,0])


	def _build_infer_graph(self):
		with tf.variable_scope("decoder", reuse=True):
			decoder_cell = self._create_rnn_cell(reuse=True)
			attention_mechanism = self._create_attention(self.encoder_outputs, self.encoder_lengths)

			attention_wrapper = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, 
				attention_mechanism=attention_mechanism, attention_layer_size=self.hidden_dim, alignment_history=True)
			attention_wrapper_init = attention_wrapper.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_states)
			output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), 
				_scope='decoder/dense', _reuse=True)

			start_tokens = tf.ones([self.batch_size], tf.int32) * data_utils.GO_ID
			end_token = data_utils.EOS_ID
			
			infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding,
				start_tokens=start_tokens, end_token=end_token)
			infer_decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_wrapper, helper=infer_helper,
				initial_state=attention_wrapper_init, output_layer=output_layer)
			decoder_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=infer_decoder,
															maximum_iterations=10)
			
			self.infer_predicted_ids = tf.expand_dims(decoder_outputs.sample_id, -1)
			self.infer_alignments = tf.transpose(final_context_state.alignment_history.stack(), [1,2,0])


	def _build_backward_graph(self):
		self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.train_logits, targets=self.decoder_targets, weights=self.mask)
		
		optimizer = tf.train.AdamOptimizer(self.lr)
		trainable_params = tf.trainable_variables()
		gradients = tf.gradients(self.loss, trainable_params)
		clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
		self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)


	def _build_summary_graph(self):
		tf.summary.scalar('loss', self.loss)
		tf.summary.scalar('perplexity', tf.exp(self.loss))
		attention_images = tf.expand_dims(self.train_alignments, -1)
		attention_images *= 255
		attention_summary = tf.summary.image("attention_images", attention_images)



	def _create_attention(self, encoder_outputs, encoder_lengths):
		if self.attention_option == "luong":
			ret = tf.contrib.seq2seq.LuongAttention(num_units=self.hidden_dim, memory=encoder_outputs, memory_sequence_length=encoder_lengths)
		elif self.attention_option == "scaled_luong":
			ret = tf.contrib.seq2seq.LuongAttention(num_units=self.hidden_dim, memory=encoder_outputs, memory_sequence_length=encoder_lengths, scale=True)
		elif self.attention_option == "bahdanau":
			ret = tf.contrib.seq2seq.BahdanauAttention(num_units=self.hidden_dim, memory=encoder_outputs, memory_sequence_length=encoder_lengths)
		elif self.attention_option == "normed_bahdanau":
			ret = tf.contrib.seq2seq.BahdanauAttention(num_units=self.hidden_dim, memory=encoder_outputs, memory_sequence_length=encoder_lengths, normalize=True)
		else:
			raise ValueError("Unknown attention option %s" % self.attention_option)
		return ret


	def train_session(self, sess, batch):
		feed_dict = {self.encoder_inputs: batch.encoder_inputs,
					  self.encoder_lengths: batch.encoder_lengths,
					  self.decoder_inputs: batch.decoder_inputs,
					  self.decoder_targets: batch.decoder_targets,
					  self.decoder_lengths: batch.decoder_lengths,
					  self.keep_prob: 0.5}
		_, loss, summary, step = sess.run([self.train_op, self.loss, self.merged, self.global_step], feed_dict=feed_dict)
		return {"loss": loss, "summary": summary, "step": step}


	def eval_session(self, sess, batch):
		feed_dict = {self.encoder_inputs: batch.encoder_inputs,
					  self.encoder_lengths: batch.encoder_lengths,
					  self.decoder_inputs: batch.decoder_inputs,
					  self.decoder_targets: batch.decoder_targets,
					  self.decoder_lengths: batch.decoder_lengths,
					  self.keep_prob: 1.0}
		loss, summary = sess.run([self.loss, self.merged], feed_dict=feed_dict)
		return {"loss": loss, "summary": summary}


	def infer_session(self, sess, batch):
		feed_dict = {self.encoder_inputs: batch.encoder_inputs,
					  self.encoder_lengths: batch.encoder_lengths,
					  self.keep_prob: 1.0}
		predicted_ids, alignments = sess.run([self.infer_predicted_ids, self.infer_alignments], feed_dict=feed_dict)
		return {"predicted_ids": predicted_ids, "alignments": alignments}


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