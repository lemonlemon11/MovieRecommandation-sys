import pandas as pd
from flask import Flask, request
from flask_restplus import Resource, Api
import recommendation
import get_movie_info


app = Flask(__name__)
api = Api(app,
		  default="Movie Recommendation System",
		  title="Movie Recommendation",
		  description="Recommend movies")

title_to_id_df = pd.read_csv('./data/title_to_id.csv', index_col=0)

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
	def get(self):
		query = request.args.get('query')
		movie_id = int(query)

		movie_you_watched, recom_other_favorite_movies, users_info = recommendation.recommend_other_favorite_movie(
			movie_id)
		movie_you_watched = process_movies([movie_you_watched])[0]
		users_info = process_users_info(users_info)
		recom_other_favorite_movies = process_movies(recom_other_favorite_movies)
		msg = {'movie_you_watched': movie_you_watched,
			   'recom_other_favorite_movies': recom_other_favorite_movies,
			   'users_info': users_info,
			   'error_code': 200}
		return msg, 200


@api.route('/movie_recommendation/same_type')
@api.param('query', 'movie id')
class SameType(Resource):
	@api.response(200, 'OK')
	@api.response(404, 'Invalid Movie ID')
	@api.doc(description="Get recommend movies with the same type")
	def get(self):
		query = request.args.get('query')
		movie_id = int(query)
		movie_you_watched, recom_same_type_movies = recommendation.recommend_same_type_movie(movie_id)
		movie_you_watched = process_movies([movie_you_watched])[0]
		recom_same_type_movies = process_movies(recom_same_type_movies)
		msg = {'movie_you_watched': movie_you_watched,
			   'recom_same_type_movies': recom_same_type_movies,
			   'error_code': 200}
		return msg, 200


@api.route('/movie_recommendation/users')
@api.param('query', 'user id')
class ForUser(Resource):
	@api.response(200, 'OK')
	@api.response(404, 'Invalid User ID')
	@api.doc(description="Get recommend movies for the user")
	def get(self):
		query = request.args.get('query')
		user_id = int(query)
		your_info, recom_movies = recommendation.recommend_your_favorite_movie(user_id)
		your_info = process_users_info([your_info])[0]
		recom_movies = process_movies(recom_movies)
		msg = {'your_info': your_info,
			   'recom_movies': recom_movies,
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
	@api.doc(description="Change movie title to movie id")
	def get(self):
		query = request.args.get('query')
		movie_title = query
		print(title_to_id_df.head())
		movie_id = movie_title_to_id(movie_title)
		msg = {'movie_id': movie_id, 'error_code': 200}
		return msg, 200


if __name__ == '__main__':
	app.run(debug=True)
