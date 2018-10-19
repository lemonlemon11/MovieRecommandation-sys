import tensorflow as tf
import pickle

# feature info: ['UserID' 'MovieID' 'Gender' 'Age' 'JobID' 'Title' 'Genres']
title_length, title_set, genres2int, features, target_values, ratings, users, \
movies, data, movies_orig, users_orig = pickle.load(open('./data/params.p', 'rb'))

title_vocb_num, genres_num, movie_id_num = pickle.load(open('./data/argument.p', 'rb'))

embed_dim = 32

# 电影ID个数
movie_id_max = movie_id_num
# 电影类型个数，有个'PADDING'
movie_categories_max = genres_num
# 电影名单词个数
movie_title_max = title_vocb_num

# 电影名长度，做词嵌入要求输入的维度是固定的，这里设置为 15
# 长度不够用空白符填充，太长则进行截断
sentences_size = title_length
# 文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = {2, 3, 4, 5}
# 文本卷积核数量
filter_num = 8

# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}


def get_inputs():
	"""
	获取movie所有特征的input
	"""
	movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
	# 电影种类中要去除'PADDING'，所以-1
	movie_categories = tf.placeholder(tf.int32, [None, movie_categories_max - 1], name="movie_categories")
	movie_titles = tf.placeholder(tf.int32, [None, 15], name="movie_titles")
	dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
	return movie_id, movie_categories, movie_titles, dropout_keep_prob


def get_movie_id_embed_layer(movie_id):
	"""
	获取movie id 的embedding
	"""
	with tf.name_scope("movie_embedding"):
		movie_id_embed_matrix = tf.Variable(tf.random_uniform([movie_id_max, embed_dim], -1, 1), name="movie_id_embed_matrix")
		movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name="movie_id_embed_layer")
	return movie_id_embed_layer


# 对电影类型的多个嵌入向量做加和
def get_movie_categories_embed_layer(movie_categories, combiner='sum'):
	"""
	定义对movie类型的embedding，同时对于一个movie的所有类型，进行combiner的组合。
	目前仅考虑combiner为sum的情况，即将该电影所有的类型进行sum求和
	"""
	with tf.name_scope("movie_categories_layers"):
		movie_categories_embed_matrix = tf.Variable(tf.random_uniform([movie_categories_max, embed_dim], -1, 1), name="movie_categories_embed_matrix")
		movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories, name="movie_categories_embed_layer")
		if combiner == "sum":
			movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keep_dims=True)
	#     elif combiner == "mean":
	return movie_categories_embed_layer


def get_movie_cnn_layer(movie_titles, dropout_keep_prob, window_sizes=[3,4,5,6]):
	"""
	对movie的title，进行卷积神经网络实现
	window_sizes:  文本卷积滑动窗口，分别滑动3,4,5, 6个单词
	"""
	# 从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
	with tf.name_scope("movie_embedding"):
		movie_title_embed_matrix = tf.Variable(tf.random_uniform([movie_title_max, embed_dim], -1, 1), name="movie_title_embed_matrix")
		movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles, name="movie_title_embed_layer")
		# 为 movie_title_embed_layer 增加一个维度
		# 在这里是添加到最后一个维度，最后一个维度是channel
		# 所以这里的channel数量是1个
		# 所以这里的处理方式和图片是一样的
		movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

	# 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
	pool_layer_lst = []
	for window_size in window_sizes:
		with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
			# [window_size, embed_dim, 1, filter_num] 表示输入的 channel 的个数是1，输出的 channel 的个数是 filter_num
			filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num], stddev=0.1), name="filter_weights")
			filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")

			# conv2d 是指用到的卷积核的大小是 [filter_height * filter_width * in_channels, output_channels]
			# 在这里卷积核会向两个维度的方向进行滑动
			# conv1d 是将卷积核向一个维度的方向进行滑动，这就是 conv1d 和 conv2d 的区别
			# strides 设置要求第一个和最后一个数字是1，四个数字的顺序要求默认是 NHWC，也就是 [batch, height, width, channels]
			# padding 设置为 VALID 其实就是不 PAD，设置为 SAME 就是让输入和输出的维度是一样的
			conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID", name="conv_layer")
			# tf.nn.bias_add 将偏差 filter_bias 加到 conv_layer 上
			# tf.nn.relu 将激活函数设置为 relu
			relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")

			# tf.nn.max_pool 的第一个参数是输入
			# 第二个参数是 max_pool 窗口的大小，每个数值表示对每个维度的窗口设置
			# 第三个参数是 strides，和 conv2d 的设置是一样的
			# 这边的池化是将上面每个卷积核的卷积结果转换为一个元素
			# 由于这里的卷积核的数量是 8 个，所以下面生成的是一个具有 8 个元素的向量
			maxpool_layer = tf.nn.max_pool(relu_layer, [1, sentences_size - window_size + 1, 1, 1], [1, 1, 1, 1], padding="VALID", name="maxpool_layer")
			pool_layer_lst.append(maxpool_layer)

	# Dropout层
	with tf.name_scope("pool_dropout"):
		# 这里最终的结果是这样的，
		# 假设卷积核的窗口是 2，卷积核的数量是 8
		# 那么通过上面的池化操作之后，生成的池化的结果是一个具有 8 个元素的向量
		# 每种窗口大小的卷积核经过池化后都会生成这样一个具有 8 个元素的向量
		# 所以最终生成的是一个 8 维的二维矩阵，它的另一个维度就是不同的窗口的数量
		# 在这里就是 2,3,4,5，那么最终就是一个 8*4 的矩阵，
		pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
		max_num = len(window_sizes) * filter_num
		# 将这个 8*4 的二维矩阵平铺成一个具有 32 个元素的一维矩阵
		pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")

		dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name="dropout_layer")
	return pool_layer_flat, dropout_layer


def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
	"""
	将movie id，movie genres， movie title的representations分别连入一个小型的神经网络
	然后将每个神经网络的输出拼接在一起，组成movie feature representation
	"""
	with tf.name_scope("movie_fc"):
		# 第一层全连接
		movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, embed_dim, name="movie_id_fc_layer", activation=tf.nn.relu)
		movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, embed_dim, name="movie_categories_fc_layer", activation=tf.nn.relu)

		# 第二层全连接
		movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  # (?, 1, 96)
		movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, 200, tf.tanh)  # (?, 1, 200)

		movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
	return movie_combine_layer, movie_combine_layer_flat
