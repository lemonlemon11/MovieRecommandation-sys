import numpy as np
import tensorflow as tf

import os
import pickle
import random

title_length, title_set, genres2int, features, target_values, ratings, users, \
movies, data, movies_orig, users_orig = pickle.load(open('./data/params.p', mode='rb'))
# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第五行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}
sentences_size = title_length
load_dir = './save_model/'
movie_feature_size = user_feature_size = 512
movie_matrix_path = './data/movie_matrix.p'
user_matrix_path = './data/user_matrix.p'


# 获取 Tensors
def get_tensors(loaded_graph):
	uid = loaded_graph.get_tensor_by_name("uid:0")
	user_gender = loaded_graph.get_tensor_by_name("user_gender:0")
	user_age = loaded_graph.get_tensor_by_name("user_age:0")
	user_job = loaded_graph.get_tensor_by_name("user_job:0")
	movie_id = loaded_graph.get_tensor_by_name("movie_id:0")
	movie_categories = loaded_graph.get_tensor_by_name("movie_categories:0")
	movie_titles = loaded_graph.get_tensor_by_name("movie_titles:0")
	targets = loaded_graph.get_tensor_by_name("targets:0")
	dropout_keep_prob = loaded_graph.get_tensor_by_name("dropout_keep_prob:0")
	# 两种不同计算预测评分的方案使用不同的name获取tensor inference
	#     inference = loaded_graph.get_tensor_by_name("inference/inference/BiasAdd:0")
	inference = loaded_graph.get_tensor_by_name(
		"inference/ExpandDims:0")  # 之前是MatMul:0 因为inference代码修改了 这里也要修改 感谢网友 @清歌 指出问题
	movie_combine_layer_flat = loaded_graph.get_tensor_by_name("movie_fc/Reshape:0")
	user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc/Reshape:0")
	return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat


# 预测指定用户对指定电影的评分
# 这部分就是对网络做正向传播，计算得到预测的评分
def rating_movie(user_id_val, movie_id_val):
	loaded_graph = tf.Graph()  #
	with tf.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		# Get Tensors from loaded model
		uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat = get_tensors(loaded_graph)  # loaded_graph

		categories = np.zeros([1, 18])
		categories[0] = movies.values[movieid2idx[movie_id_val]][2]

		titles = np.zeros([1, sentences_size])
		titles[0] = movies.values[movieid2idx[movie_id_val]][1]

		feed = {
			uid: np.reshape(users.values[user_id_val - 1][0], [1, 1]),
			user_gender: np.reshape(users.values[user_id_val - 1][1], [1, 1]),
			user_age: np.reshape(users.values[user_id_val - 1][2], [1, 1]),
			user_job: np.reshape(users.values[user_id_val - 1][3], [1, 1]),
			movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
			movie_categories: categories,  # x.take(6,1)
			movie_titles: titles,  # x.take(5,1)
			dropout_keep_prob: 1}

		# Get Prediction
		inference_val = sess.run([inference], feed)

		return inference_val


# 生成movie特征矩阵，将训练好的电影特征组合成电影特征矩阵并保存到本地
# 对每个电影进行正向传播
def save_movie_feature_matrix():
	loaded_graph = tf.Graph()  #
	movie_matrics = []
	with tf.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		# Get Tensors from loaded model
		uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat = get_tensors(loaded_graph)  # loaded_graph

		for item in movies.values:
			categories = np.zeros([1, 18])
			categories[0] = item.take(2)

			titles = np.zeros([1, sentences_size])
			titles[0] = item.take(1)

			feed = {
				movie_id: np.reshape(item.take(0), [1, 1]),
				movie_categories: categories,  # x.take(6,1)
				movie_titles: titles,  # x.take(5,1)
				dropout_keep_prob: 1}

			movie_combine_layer_flat_val = sess.run([movie_combine_layer_flat], feed)
			movie_matrics.append(movie_combine_layer_flat_val)

	pickle.dump((np.array(movie_matrics).reshape(-1, 200)), open(movie_matrix_path, 'wb'))


# 生成user特征矩阵
# 将训练好的用户特征组合成用户特征矩阵并保存到本地
# 对每个用户进行正向传播
def save_user_feature_matrix():
	loaded_graph = tf.Graph()  #
	users_matrics = []
	with tf.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		# Get Tensors from loaded model
		uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat = get_tensors(loaded_graph)  # loaded_graph

		for item in users.values:
			feed = {
				uid: np.reshape(item.take(0), [1, 1]),
				user_gender: np.reshape(item.take(1), [1, 1]),
				user_age: np.reshape(item.take(2), [1, 1]),
				user_job: np.reshape(item.take(3), [1, 1]),
				dropout_keep_prob: 1}

			user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)
			users_matrics.append(user_combine_layer_flat_val)

	pickle.dump((np.array(users_matrics).reshape(-1, 200)), open(user_matrix_path, 'wb'))


def load_feature_matrix(path):
	if os.path.exists(path):
		pass
	else:
		if path == movie_matrix_path:
			save_movie_feature_matrix()
		else:
			save_user_feature_matrix()
	return pickle.load(open(path, 'rb'))



# 使用电影特征矩阵推荐同类型的电影
# 思路是计算指定电影的特征向量与整个电影特征矩阵的余弦相似度，
# 取相似度最大的top_k个，
# ToDo: 加入随机选择，保证每次的推荐稍微不同
def recommend_same_type_movie(movie_id_val, top_k=20):
	movie_matrics = load_feature_matrix(movie_matrix_path)
	loaded_graph = tf.Graph()  #
	with tf.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keep_dims=True))
		normalized_movie_matrics = movie_matrics / norm_movie_matrics

		# 推荐同类型的电影
		probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
		probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
		sim = (probs_similarity.eval())
		#     results = (-sim[0]).argsort()[0:top_k]
		#     print(results)

		# print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))
		# print("以下是给您的推荐：")
		p = np.squeeze(sim)
		p[np.argsort(p)[:-top_k]] = 0
		p = p / np.sum(p)
		results = set()
		while len(results) != 5:
			c = np.random.choice(3883, 1, p=p)[0]
			results.add(c)
		recom_movies = []
		movie_you_watched = list(movies_orig[movieid2idx[movie_id_val]])
		for val in results:
			# print(val)
			# print(movies_orig[val])
			recom_movies.append(list(movies_orig[val]))

		# return results
		return movie_you_watched, recom_movies


# 给定指定用户，推荐其喜欢的电影
# 思路是使用用户特征向量与电影特征矩阵计算所有电影的评分，
# 取评分最高的top_k个，
# ToDo 加入随机选择
def recommend_your_favorite_movie(user_id_val, top_k=20):
	user_matrics = load_feature_matrix(user_matrix_path)
	movie_matrics = load_feature_matrix(movie_matrix_path)
	loaded_graph = tf.Graph()  #
	with tf.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		# 推荐您喜欢的电影
		probs_embeddings = (user_matrics[user_id_val - 1]).reshape([1, 200])

		probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
		sim = (probs_similarity.eval())
		#     print(sim.shape)
		#     results = (-sim[0]).argsort()[0:top_k]
		#     print(results)

		#     sim_norm = probs_norm_similarity.eval()
		#     print((-sim_norm[0]).argsort()[0:top_k])

		# print("以下是给您的推荐：")
		p = np.squeeze(sim)
		p[np.argsort(p)[:-top_k]] = 0
		p = p / np.sum(p)
		results = set()
		while len(results) != 5:
			c = np.random.choice(3883, 1, p=p)[0]
			results.add(c)
		recom_movies = []
		for val in results:
			# print(val)
			# print(movies_orig[val])
			recom_movies.append(list(movies_orig[val]))
		your_info = users_orig[user_id_val - 1]
		# return results
		return your_info, recom_movies


# 看过这个电影的人还可能（喜欢）哪些电影
# 首先选出喜欢某个电影的top_k个人，得到这几个人的用户特征向量
# 然后计算这几个人对所有电影的评分
# 选择每个人评分最高的电影作为推荐
# ToDo 加入随机选择
def recommend_other_favorite_movie(movie_id_val, top_k=20):
	user_matrics = load_feature_matrix(user_matrix_path)
	movie_matrics = load_feature_matrix(movie_matrix_path)
	loaded_graph = tf.Graph()  #
	with tf.Session(graph=loaded_graph) as sess:  #
		# Load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
		probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(user_matrics))
		favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0][-top_k:]
		#     print(normalized_users_matrics.eval().shape)
		#     print(probs_user_favorite_similarity.eval()[0][favorite_user_id])
		#     print(favorite_user_id.shape)

		# print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))

		# print("喜欢看这个电影的人是：{}".format(users_orig[favorite_user_id - 1]))
		users_info = users_orig[favorite_user_id - 1]
		probs_users_embeddings = (user_matrics[favorite_user_id - 1]).reshape([-1, 200])
		probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
		sim = (probs_similarity.eval())
		#     results = (-sim[0]).argsort()[0:top_k]
		#     print(results)

		#     print(sim.shape)
		#     print(np.argmax(sim, 1))
		p = np.argmax(sim, 1)
		# print("喜欢看这个电影的人还喜欢看：")

		results = set()
		while len(results) != 5:
			c = p[random.randrange(top_k)]
			results.add(c)
		recom_movies = []
		movie_you_watched = list(movies_orig[movieid2idx[movie_id_val]])
		for val in (results):
			# print(val)
			# print(movies_orig[val])
			recom_movies.append(list(movies_orig[val]))

		# return results
		return movie_you_watched, recom_movies, users_info


# print(recommend_your_favorite_movie(222))
# test every recommendation functions here

# 预测给定user对给定movie的评分
# prediction_rating = rating_movie(user_id=123, movie_id=1234)
# print('for user:123, predicting the rating for movie:1234', prediction_rating)

# 生成user和movie的特征矩阵，并存储到本地
# save_movie_feature_matrix()
# save_user_feature_matrix()

# 对给定的电影，推荐相同类型的其他top k 个电影
# results = recommend_same_type_movie(movie_id_val=666, top_k=5)
# print(results)

# print(results)
# 对给定用户，推荐其可能喜欢的top k个电影
# results = recommend_your_favorite_movie(user_id_val=222, top_k=20)
# print(results)
# print(results)
# 看过这个电影的人还可能喜欢看那些电影
# print(recommend_other_favorite_movie(movie_id_val=666, top_k=5))

# title_length：Title字段的长度（15）
# title_set：Title文本的集合
# genres2int：电影类型转数字的字典
# features：是输入X
# targets_values：是学习目标y
# ratings：评分数据集的Pandas对象
# users：用户数据集的Pandas对象
# movies：电影数据的Pandas对象
# data：三个数据集组合在一起的Pandas对象
# movies_orig：没有做数据处理的原始电影数据
# users_orig：没有做数据处理的原始用户数据
