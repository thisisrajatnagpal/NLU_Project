import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import collections as coll

import tensorflow as tf

from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, f1_score

GCNLayer(self, gcn_in, in_dim, gcn_dim, batch_size, max_nodes, max_labels, adj_ind, adj_data, w_gating=True, num_layers=1, name="GCN"):
		"""
		GCN Layer Implementation
		Parameters
		----------
		gcn_in:		Input to GCN Layer
		in_dim:		Dimension of input to GCN Layer 
		gcn_dim:	Hidden state dimension of GCN
		batch_size:	Batch size
		max_nodes:	Maximum number of nodes in graph
		max_labels:	Maximum number of edge labels
		adj_ind:	Adjacency matrix indices
		adj_data:	Adjacency matrix data (all 1's)
		w_gating:	Whether to include gating in GCN
		num_layers:	Number of GCN Layers
		name 		Name of the layer (used for creating variables, keep it different for different layers)
		Returns
		-------
		out		List of output of different GCN layers with first element as input itself, i.e., [gcn_in, gcn_layer1_out, gcn_layer2_out ...]
		"""

		out = []
		out.append(gcn_in)

		for layer in range(num_layers):
			gcn_in    = out[-1]				# out contains the output of all the GCN layers, intitally contains input to first GCN Layer
			if len(out) > 1: in_dim = gcn_dim 		# After first iteration the in_dim = gcn_dim

			with tf.name_scope('%s-%d' % (name,layer)):
				act_sum = tf.zeros([batch_size, max_nodes, gcn_dim])
				for lbl in range(max_labels):

					# Defining the layer and label specific parameters
					with tf.variable_scope('label-%d_name-%s_layer-%d' % (lbl, name, layer)) as scope:
						w_in   = tf.get_variable('w_in',  	[in_dim, gcn_dim], initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
						w_out  = tf.get_variable('w_out', 	[in_dim, gcn_dim], initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
						w_loop = tf.get_variable('w_loop', 	[in_dim, gcn_dim], initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
						b_in   = tf.get_variable('b_in',   	initializer=np.zeros([1, gcn_dim]).astype(np.float32),	regularizer=self.regularizer)
						b_out  = tf.get_variable('b_out',  	initializer=np.zeros([1, gcn_dim]).astype(np.float32),	regularizer=self.regularizer)

						if w_gating:
							w_gin  = tf.get_variable('w_gin',   [in_dim, 1], initializer=tf.contrib.layers.xavier_initializer(),	regularizer=self.regularizer)
							w_gout = tf.get_variable('w_gout',  [in_dim, 1], initializer=tf.contrib.layers.xavier_initializer(),	regularizer=self.regularizer)
							w_gloop = tf.get_variable('w_gloop',[in_dim, 1], initializer=tf.contrib.layers.xavier_initializer(),	regularizer=self.regularizer)
							b_gin  = tf.get_variable('b_gin',   initializer=np.zeros([1]).astype(np.float32),	regularizer=self.regularizer)
							b_gout = tf.get_variable('b_gout',  initializer=np.zeros([1]).astype(np.float32),	regularizer=self.regularizer)

					# Activation from in-edges
					with tf.name_scope('in_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
						inp_in  = tf.tensordot(gcn_in, w_in, axes=[2,0]) + tf.expand_dims(b_in, axis=0)

						def map_func1(i):
							adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i], [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
							adj_mat = tf.sparse_transpose(adj_mat)
							return tf.sparse_tensor_dense_matmul(adj_mat, inp_in[i])
						in_t = tf.map_fn(map_func1, tf.range(batch_size), dtype=tf.float32)

						if self.p.dropout != 1.0: in_t = tf.nn.dropout(in_t, keep_prob=self.p.dropout)

						if w_gating:
							inp_gin = tf.tensordot(gcn_in, w_gin, axes=[2,0]) + tf.expand_dims(b_gin, axis=0)

							def map_func2(i):
								adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i], [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
								adj_mat = tf.sparse_transpose(adj_mat)
								return tf.sparse_tensor_dense_matmul(adj_mat, inp_gin[i])
							in_gate = tf.map_fn(map_func2, tf.range(batch_size), dtype=tf.float32)
							in_gsig = tf.sigmoid(in_gate)
							in_act  = in_t * in_gsig
						else:
							in_act   = in_t

					# Activation from out-edges
					with tf.name_scope('out_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
						inp_out  = tf.tensordot(gcn_in, w_out, axes=[2,0]) + tf.expand_dims(b_out, axis=0)

						def map_func3(i):
							adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i], [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
							return tf.sparse_tensor_dense_matmul(adj_mat, inp_out[i])
						out_t = tf.map_fn(map_func3, tf.range(batch_size), dtype=tf.float32)
						if self.p.dropout != 1.0: out_t    = tf.nn.dropout(out_t, keep_prob=self.p.dropout)

						if w_gating:
							inp_gout = tf.tensordot(gcn_in, w_gout, axes=[2,0]) + tf.expand_dims(b_gout, axis=0)
							def map_func4(i):
								adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i], [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
								return tf.sparse_tensor_dense_matmul(adj_mat, inp_gout[i])
								
							out_gate = tf.map_fn(map_func4, tf.range(batch_size), dtype=tf.float32)
							out_gsig = tf.sigmoid(out_gate)
							out_act  = out_t * out_gsig
						else:
							out_act = out_t

					# Activation from self-loop
					with tf.name_scope('self_loop'):
						inp_loop  = tf.tensordot(gcn_in, w_loop,  axes=[2,0])
						if self.p.dropout != 1.0: inp_loop  = tf.nn.dropout(inp_loop, keep_prob=self.p.dropout)

						if w_gating:
							inp_gloop = tf.tensordot(gcn_in, w_gloop, axes=[2,0])
							loop_gsig = tf.sigmoid(inp_gloop)
							loop_act  = inp_loop * loop_gsig
						else:
							loop_act = inp_loop

					# Aggregating activations
					act_sum += in_act + out_act + loop_act

				gcn_out = tf.nn.relu(act_sum)
				out.append(gcn_out)

return out