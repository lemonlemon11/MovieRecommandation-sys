import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import pickle

import matplotlib.pyplot as plt
import time
import datetime

import movie_nn
import user_nn

tf.reset_default_graph()
train_graph = tf.Graph()

features = pickle.load(open('./data/features.p', 'rb'))
target_values = pickle.load(open('./data/targets.p', 'rb'))
# title_length, title_set, genres2int, features, target_values,ratings, users,\
#  movies, data, movies_orig, users_orig = pickle.load(open('params.p', 'rb'))


# 超参
# Number of Epochs
num_epochs = 0
# Batch Size
batch_size = 256
dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 50
# 对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
combiner = "sum"
title_length = 15
save_dir = './save_model/'


def get_targets():
	targets = tf.placeholder(tf.int32, [None, 1], name="targets")
	# lr = tf.placeholder(tf.float32, name="LearningRate")
	return targets


# 自定义获取 batch 的方法
def get_batches(Xs, ys, batch_size):
	for start in range(0, len(Xs), batch_size):
		end = min(start + batch_size, len(Xs))
		yield Xs[start:end], ys[start:end]


with train_graph.as_default():
	global_step = tf.Variable(0, name='global_step', trainable=True)
	targets = get_targets()

	# 获取输入占位符
	uid, user_gender, user_age, user_job = user_nn.get_inputs()
	movie_id, movie_categories, movie_titles, dropout_keep_prob = movie_nn.get_inputs()
	# 获取User的4个嵌入向量
	uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = user_nn.get_user_embedding(uid, user_gender,
																									   user_age,
																									   user_job)
	# 得到用户特征
	user_combine_layer, user_combine_layer_flat = user_nn.get_user_feature_layer(uid_embed_layer, gender_embed_layer,
																				 age_embed_layer, job_embed_layer)
	# 获取电影ID的嵌入向量
	movie_id_embed_layer = movie_nn.get_movie_id_embed_layer(movie_id)
	# 获取电影类型的嵌入向量
	movie_categories_embed_layer = movie_nn.get_movie_categories_embed_layer(movie_categories, combiner)
	# 获取电影名的特征向量
	pool_layer_flat, dropout_layer = movie_nn.get_movie_cnn_layer(movie_titles, dropout_keep_prob)
	# 得到电影特征
	movie_combine_layer, movie_combine_layer_flat = movie_nn.get_movie_feature_layer(movie_id_embed_layer,
																					 movie_categories_embed_layer,
																					 dropout_layer)
	# 计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
	# tensorflow 的 name_scope 指定了 tensor 范围，方便我们后面调用，通过指定 name_scope 来调用其中的 tensor
	with tf.name_scope("inference"):
		# 直接将用户特征矩阵和电影特征矩阵相乘得到得分，最后要做的就是对这个得分进行回归
		inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
		inference = tf.expand_dims(inference, axis=1)

	with tf.name_scope("loss"):
		# MSE损失，将计算值回归到评分
		cost = tf.losses.mean_squared_error(targets, inference)
		# 将每个维度的 cost 相加，计算它们的平均值
		loss = tf.reduce_mean(cost)
	# 优化损失
	#     train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
	# optimizer = tf.train.AdamOptimizer()
	# gradients = optimizer.compute_gradients(loss)  # cost
	# train_op = optimizer.apply_gradients(gradients, global_step=global_step)
	train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

losses = {'train': [], 'test': []}

with tf.Session(graph=train_graph) as sess:
	# 搜集数据给tensorBoard用
	# Keep track of gradient values and sparsity
	# grad_summaries = []
	# for g, v in gradients:
	# 	if g is not None:
	# 		grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
	# 		# tf.nn.zero_fraction 用于计算矩阵中 0 所占的比重，也就是计算矩阵的稀疏程度
	# 		sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')), tf.nn.zero_fraction(g))
	# 		grad_summaries.append(grad_hist_summary)
	# 		grad_summaries.append(sparsity_summary)
	# grad_summaries_merged = tf.summary.merge(grad_summaries)

	# Output directory for models and summaries
	timestamp = str(int(time.time()))
	out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
	print(f"Writing to {out_dir}\n")

	# Summaries for loss and accuracy
	loss_summary = tf.summary.scalar("loss", loss)

	# Train Summaries
	# train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
	train_summary_op = tf.summary.merge([loss_summary])
	train_summary_dir = os.path.join(out_dir, "summaries", "train")
	train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

	# Inference summaries
	inference_summary_op = tf.summary.merge([loss_summary])
	inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
	inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)

	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	for epoch_i in range(num_epochs):

		# 将数据集分成训练集和测试集，随机种子不固定
		train_X, test_X, train_y, test_y = train_test_split(features, target_values, test_size=0.2, random_state=0)

		train_batches = get_batches(train_X, train_y, batch_size)
		test_batches = get_batches(test_X, test_y, batch_size)

		# 训练的迭代，保存训练损失
		for batch_i in range(len(train_X) // batch_size):
			x, y = next(train_batches)

			categories = np.zeros([batch_size, 18])
			for i in range(batch_size):
				categories[i] = x.take(6, 1)[i]

			titles = np.zeros([batch_size, title_length])
			for i in range(batch_size):
				titles[i] = x.take(5, 1)[i]

			feed = {
				uid: np.reshape(x.take(0, 1), [batch_size, 1]),
				user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
				user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
				user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
				movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
				movie_categories: categories,  # x.take(6,1)
				movie_titles: titles,  # x.take(5,1)
				targets: np.reshape(y, [batch_size, 1]),
				dropout_keep_prob: dropout_keep
			}

			step, train_loss, summaries, _ = sess.run([global_step, loss, train_summary_op, train_op], feed)  # cost
			losses['train'].append(train_loss)
			train_summary_writer.add_summary(summaries, step)  #

			# Show every <show_every_n_batches> batches
			if batch_i % show_every_n_batches == 0:
				time_str = datetime.datetime.now().isoformat()
				print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
					time_str,
					epoch_i,
					batch_i,
					(len(train_X) // batch_size),
					train_loss))

		# 使用测试数据的迭代
		for batch_i in range(len(test_X) // batch_size):
			x, y = next(test_batches)

			categories = np.zeros([batch_size, 18])
			for i in range(batch_size):
				categories[i] = x.take(6, 1)[i]

			titles = np.zeros([batch_size, title_length])
			for i in range(batch_size):
				titles[i] = x.take(5, 1)[i]

			feed = {
				uid: np.reshape(x.take(0, 1), [batch_size, 1]),
				user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
				user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
				user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
				movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
				movie_categories: categories,  # x.take(6,1)
				movie_titles: titles,  # x.take(5,1)
				targets: np.reshape(y, [batch_size, 1]),
				dropout_keep_prob: 1
			}

			step, test_loss, summaries = sess.run([global_step, loss, inference_summary_op], feed)  # cost

			# 保存测试损失
			losses['test'].append(test_loss)
			inference_summary_writer.add_summary(summaries, step)  #

			time_str = datetime.datetime.now().isoformat()
			if batch_i % show_every_n_batches == 0:
				print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
					time_str,
					epoch_i,
					batch_i,
					(len(test_X) // batch_size),
					test_loss))

	# Save Model
	saver.save(sess, save_dir)
	print('Model Trained and Saved')

# plt.plot(losses['train'], label='training loss')
# plt.legend()
# plt.show()
# plt.plot(losses['test'], label='test loss')
# plt.legend()
# plt.show()
