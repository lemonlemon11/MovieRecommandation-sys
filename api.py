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


def movie_title_to_id(movie_title):
	movie_id = title_to_id_df.loc[movie_title]['MovieID']
	return int(movie_id)


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
