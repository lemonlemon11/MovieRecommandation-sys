import pandas as pd
from flask import Flask
from flask_restplus import Resource, Api, fields, reqparse
import recommendation
import get_movie_info

from functools import wraps
from time import time

from flask import request
from flask_restplus import abort
from itsdangerous import SignatureExpired, JSONWebSignatureSerializer, BadSignature


app = Flask(__name__)
api = Api(app,
		  security='API-KEY',
		  default="Movie Recommendation System",
		  title="Movie Recommendation",
		  description="Recommend movies")

user_age_dict = {1: "Under 18",
				 18: "18-24",
				 25: "25-34",
				 35: "35-44",
				 45: "45-49",
				 50: "50-55",
				 56: "56+"}

user_job_dict = {0: "other or not specified",
				 1: "academic/educator",
				 2: "artist",
				 3: "clerical/admin",
				 4: "college/grad student",
				 5: "customer service",
				 6: "doctor/health care",
				 7: "executive/managerial",
				 8: "farmer",
				 9: "homemaker",
				 10: "K-12 student",
				 11: "lawyer",
				 12: "programmer",
				 13: "retired",
				 14: "sales/marketing",
				 15: "scientist",
				 16: "self-employed",
				 17: "technician/engineer",
				 18: "tradesman/craftsman",
				 19: "unemployed",
				 20: "writer"}

title_to_id_df = pd.read_csv('./data/title_to_id.csv', index_col=0)


# =========================== authentication ===========================
credential_model = api.model('credential', {
	'username': fields.String,
})

credential_parser = reqparse.RequestParser()
credential_parser.add_argument('username', type=str)
# credential_parser.add_argument('password', type=str)

# userList = [{'steven':'111'}, {"neo":'111'}, {'krist':'111'}, {"tracy":'111'}]
# userList = ['steven', "neo", 'krist', "tracy"]
userList = {'steven':'111', "neo":'111', 'krist':'111', "tracy":'111','test':'test','admin':'admin'}
test = ""
user_auth = {}


class AuthenticationToken:
	def __init__(self, secret_key, expires_in):
		self.secret_key = secret_key
		self.expires_in = expires_in
		self.serializer = JSONWebSignatureSerializer(secret_key)

	def generate_token(self, username):
		info = {
			'username': username,
			'creation_time': time()
		}

		token = self.serializer.dumps(info)
		return token.decode()

	def validate_token(self, token):
		info = self.serializer.loads(token.encode())

		if time() - info['creation_time'] > self.expires_in:
			raise SignatureExpired("The Token has been expired; get a new token")

		return info['username']


SECRET_KEY = "A SECRET KEY; USUALLY A VERY LONG RANDOM STRING"
expires_in = 600
auth = AuthenticationToken(SECRET_KEY, expires_in)


def requires_auth(f):
	@wraps(f)
	def decorated(*args, **kwargs):
		username = test
		if not username:
			abort(404, 'Authentication token is missing')
		if username not in userList:
			abort(401, "Authentication failed")

		try:
			user = auth.validate_token(user_auth[username])
		except SignatureExpired as e:
			abort(404, e.message)
		except BadSignature as e:
			abort(404, e.message)

		return f(*args, **kwargs)

	return decorated
#-------new function-------
@api.route('/add_user')
@api.param('account', 'user name')
@api.param('pwd', 'pwd')
class Add_user(Resource):
	def get(self):
		username = request.args.get('account')
		pwd = request.args.get('pwd')
		userList[username]=pwd

		print(userList)
		return {"url":"login.html"}
#-------new function-------

@api.route('/token')
class Token(Resource):
	@api.response(200, 'OK')
	@api.response(404, 'Authorization has been refused')
	@api.doc(description="Generates a authentication token")
	@api.expect(credential_parser, validate=True)
	def get(self):
		args = credential_parser.parse_args()
		username = request.args.get('username')
		password = request.args.get('password')
		print(username, password)
		global test
		test = username
		if username in userList.keys() and userList[username]==password:
			user_auth[test] = auth.generate_token(test)
			return  {'code':'200'}

		return {'code': '400'}


# ======================================================================


def movie_title_to_id(movie_title):
	movie_id = title_to_id_df.loc[movie_title]['MovieID']
	return int(movie_id)


# change users_info from [[], [],...] to [{}, {},...]
def process_users_info(users_info):
	processed_users_info = []
	for user in users_info:
		user_info_dict = {}
		user_info_dict['user_id'] = user[0]
		user_info_dict['gender'] = user[1]
		user_info_dict['age'] = user_age_dict[user[2]]
		user_info_dict['job'] = user_job_dict[user[3]]
		processed_users_info.append(user_info_dict)
	return processed_users_info


# change movies from [[], [],...] to [{}, {},...]
def process_movies(movies):
	processed_movies = []
	for movie in movies:
		movie_info_dict = {}
		movie_info_dict['movie_id'] = movie[0]
		movie_info_dict['movie_name'] = movie[1]
		movie_info_dict['type'] = movie[2]
		processed_movies.append(movie_info_dict)
	return processed_movies


@api.route('/movie_recommendation/other_favorite')
@api.param('query', 'movie id')
class OtherFavorite(Resource):
	@api.response(200, 'OK')
	@api.response(404, 'Invalid Movie ID')
	@api.doc(description="Get recommend favorite movies of others who watched this film as well")
	@requires_auth

	def get(self):
		query = request.args.get('query')
		movie_id = int(query)

		movie_you_watched, recom_other_favorite_movies, users_info = recommendation.recommend_other_favorite_movie(
			movie_id)
		movie_you_watched = process_movies([movie_you_watched])[0]
		users_info = process_users_info(users_info)
		recom_other_favorite_movies = process_movies(recom_other_favorite_movies)

		moive_name_list = []
		for data in recom_other_favorite_movies:
			moive_name_list.append(data['movie_name'])

		movie_info = get_movie_info.get_movie_info(moive_name_list)

		total_info_list = {}

		info_length = len(movie_info)

		data_list = []
		for idx in range(0, info_length):
			info_dict = {}
			info_dict['id'] = recom_other_favorite_movies[idx]['movie_id']
			info_dict['movie_name'] = recom_other_favorite_movies[idx]['movie_name']
			info_dict['type'] = recom_other_favorite_movies[idx]['type']
			info_dict['img_url'] = movie_info[idx]['poster'] if movie_info[idx][
																	'poster'] is not None else 'static/images/none.png'
			info_dict['summary'] = movie_info[idx]['summary'] if movie_info[idx]['poster'] is not None else 'None'
			info_dict['time'] = movie_info[idx]['time'] if movie_info[idx]['poster'] is not None else 'None'
			info_dict['director'] = movie_info[idx]['director'] if movie_info[idx]['poster'] is not None else 'None'
			info_dict['rate'] = movie_info[idx]['rate'] if movie_info[idx]['poster'] is not None else 'None'
			start_str = ''

			if movie_info[idx]['stars'] is not None:
				for star in movie_info[idx]['stars']:
					start_str += star + ' '
				info_dict['stars'] = start_str
			else:
				info_dict['stars'] = 'None'
			data_list.append(info_dict)
		total_info_list['data'] = data_list
		total_info_list['draw'] = 1
		total_info_list['recordsTotal'] = info_length
		total_info_list['recordsFiltered'] = info_length



		msg = {'movie_you_watched': movie_you_watched,
			   'recom_other_favorite_movies': total_info_list,
			   'users_info': users_info,
			   'error_code': 200}
		return msg, 200


@api.route('/movie_recommendation/same_type')
@api.param('query', 'movie id')
class SameType(Resource):
	@api.response(200, 'OK')
	@api.response(404, 'Invalid Movie ID')
	@api.doc(description="Get recommend movies with the same type")
	@requires_auth

	def get(self):
		query = request.args.get('query')
		movie_id = int(query)
		movie_you_watched, recom_same_type_movies = recommendation.recommend_same_type_movie(movie_id)
		movie_you_watched = process_movies([movie_you_watched])[0]
		recom_same_type_movies = process_movies(recom_same_type_movies)

		moive_name_list = []
		for data in recom_same_type_movies:
			moive_name_list.append(data['movie_name'])

		print(recom_same_type_movies)
		movie_info = get_movie_info.get_movie_info(moive_name_list)

		total_info_list = {}

		info_length = len(movie_info)

		data_list = []
		for idx in range(0, info_length):
			info_dict = {}
			info_dict['id'] = recom_same_type_movies[idx]['movie_id']
			info_dict['movie_name'] = recom_same_type_movies[idx]['movie_name']
			info_dict['type'] = recom_same_type_movies[idx]['type']
			info_dict['img_url'] = movie_info[idx]['poster'] if movie_info[idx][
																	'poster'] is not None else 'None'
			info_dict['summary'] = movie_info[idx]['summary'] if movie_info[idx]['poster'] is not None else 'None'
			info_dict['time'] = movie_info[idx]['time'] if movie_info[idx]['poster'] is not None else 'None'
			info_dict['director'] = movie_info[idx]['director'] if movie_info[idx]['poster'] is not None else 'None'
			info_dict['rate'] = movie_info[idx]['rate'] if movie_info[idx]['poster'] is not None else 'None'
			start_str = ''

			if movie_info[idx]['stars'] is not None:
				for star in movie_info[idx]['stars']:
					start_str += star + ' '
				info_dict['stars'] = start_str
			else:
				info_dict['stars'] = 'None'
			data_list.append(info_dict)
		total_info_list['data'] = data_list
		total_info_list['draw'] = 1
		total_info_list['recordsTotal'] = info_length
		total_info_list['recordsFiltered'] = info_length


		msg = {'movie_you_watched': movie_you_watched,
			   'recom_same_type_movies': total_info_list,
			   'error_code': 200}
		return msg, 200


@api.route('/movie_recommendation/users')
@api.param('query', 'user id')
class ForUser(Resource):
	@api.response(200, 'OK')
	@api.response(404, 'Invalid User ID')
	@api.doc(description="Get recommend movies for the user")
	@requires_auth
	def get(self):
		query = request.args.get('query')
		user_id = int(query)
		your_info, recom_movies = recommendation.recommend_your_favorite_movie(user_id)
		your_info = process_users_info([your_info])[0]
		recom_movies = process_movies(recom_movies)

		moive_name_list = []
		for data in recom_movies:
			moive_name_list.append(data['movie_name'])

		movie_info = get_movie_info.get_movie_info(moive_name_list)
		total_info_list = {}

		info_length = len(movie_info)
		data_list = []
		for idx in range(0, info_length):
			info_dict = {}
			info_dict['id'] = recom_movies[idx]['movie_id']
			info_dict['movie_name'] = recom_movies[idx]['movie_name']
			info_dict['type'] = recom_movies[idx]['type']
			info_dict['img_url'] = movie_info[idx]['poster'] if movie_info[idx][
																	'poster'] is not None else 'static/images/none.png'
			info_dict['summary'] = movie_info[idx]['summary'] if movie_info[idx]['poster'] is not None else 'None'
			info_dict['time'] = movie_info[idx]['time'] if movie_info[idx]['poster'] is not None else 'None'
			info_dict['director'] = movie_info[idx]['director'] if movie_info[idx]['poster'] is not None else 'None'
			info_dict['rate'] = movie_info[idx]['rate'] if movie_info[idx]['poster'] is not None else 'None'
			start_str = ''

			if movie_info[idx]['stars'] is not None:
				for star in movie_info[idx]['stars']:
					start_str += star + ' '
				info_dict['stars'] = start_str
			else:
				info_dict['stars'] = 'None'
			data_list.append(info_dict)
		total_info_list['data'] = data_list
		total_info_list['draw'] = 1
		total_info_list['recordsTotal'] = info_length
		total_info_list['recordsFiltered'] = info_length

		msg = {'your_info': your_info,
			   'recom_movies': total_info_list,
			   'error_code': 200}
		return msg, 200


@api.route('/movie_recommendation/movie_info')
@api.param('query', 'list of titles')
class MovieInfo(Resource):
	@api.response(200, 'OK')
	@api.response(404, 'Invalid Movie Titles')
	@api.doc(description="Get intro of the movie")
	def get(self):
		query = request.args.get('query')
		movie_titles = query
		movies_info = get_movie_info.movie(movie_titles).get_movie_info()
		msg = {'movies_info': movies_info,
			   'error_code': 200}
		return msg, 200


@api.route('/movie_recommendation/movie_title_to_id')
@api.param('query', 'movie title')
class TitleToID(Resource):
	@api.response(200, 'OK')
	@api.response(404, 'Invalid Movie Title')
	@api.doc(description="Get intro of the movie")
	def get(self):
		query = request.args.get('query')
		movie_title = query
		print(title_to_id_df.head())
		movie_id = movie_title_to_id(movie_title)
		msg = {'movie_id': movie_id, 'error_code': 200}
		return msg, 200




if __name__ == '__main__':
	app.run(port=9999,debug=True)
