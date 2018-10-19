import tensorflow as tf
import pickle


features = pickle.load(open('./data/features.p', mode='rb'))
# 嵌入矩阵的维度
embed_dim = 32
# 下面之所以要 +1 是因为编号和实际数量之间是差 1 的
# 用户ID个数
uid_max = max(features.take(0, 1)) + 1  # 6040
# 性别个数
gender_max = max(features.take(2, 1)) + 1  # 1 + 1 = 2
# 年龄类别个数
age_max = max(features.take(3, 1)) + 1  # 6 + 1 = 7
# 职业个数
job_max = max(features.take(4, 1)) + 1  # 20 + 1 = 21


def get_inputs():
	uid = tf.placeholder(tf.int32, [None, 1], name="uid")
	user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
	user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
	user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")
	return uid, user_gender, user_age, user_job


def get_user_embedding(uid, user_gender, user_age, user_job):
	with tf.name_scope("user_embedding"):
		# 下面的操作和情感分析项目中的单词转换为词向量的操作本质上是一样的
		# 用户的特征维度设置为 32
		# 先初始化一个非常大的用户矩阵
		# tf.random_uniform 的第二个参数是初始化的最小值，这里是-1，第三个参数是初始化的最大值，这里是1
		uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max, embed_dim], -1, 1), name="uid_embed_matrix")
		# 根据指定用户ID找到他对应的嵌入层
		uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name="uid_embed_layer")

		# 性别的特征维度设置为 32
		gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max, embed_dim], -1, 1), name="gender_embed_matrix")
		gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name="gender_embed_layer")

		# 年龄的特征维度设置为 32
		age_embed_matrix = tf.Variable(tf.random_uniform([age_max, embed_dim], -1, 1), name="age_embed_matrix")
		age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")

		# 职业的特征维度设置为 32
		job_embed_matrix = tf.Variable(tf.random_uniform([job_max, embed_dim], -1, 1), name="job_embed_matrix")
		job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name="job_embed_layer")
	# 返回产生的用户数据数据
	return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer


def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
	with tf.name_scope("user_fc"):
		# 第一层全连接
		# tf.layers.dense 的第一个参数是输入，第二个参数是层的单元的数量
		uid_fc_layer = tf.layers.dense(uid_embed_layer, embed_dim, name="uid_fc_layer", activation=tf.nn.relu)
		gender_fc_layer = tf.layers.dense(gender_embed_layer, embed_dim, name="gender_fc_layer", activation=tf.nn.relu)
		age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim, name="age_fc_layer", activation=tf.nn.relu)
		job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, name="job_fc_layer", activation=tf.nn.relu)

		# 第二层全连接
		# 将上面的每个分段组成一个完整的全连接层
		user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)  # (?, 1, 128)
		# 验证上面产生的 tensorflow 是否是 128 维度的
		# print(user_combine_layer.shape)
		# tf.contrib.layers.fully_connected 的第一个参数是输入，第二个参数是输出
		# 这里的输入是user_combine_layer，输出是200，是指每个用户有200个特征
		# 相当于是一个200个分类的问题，每个分类的可能性都会输出，在这里指的就是每个特征的可能性
		user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)  # (?, 1, 200)

		user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])
	return user_combine_layer, user_combine_layer_flat



