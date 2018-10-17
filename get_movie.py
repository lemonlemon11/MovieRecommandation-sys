import pandas as pd


def get_movie(year, genre):
	movies_title = ['MovieID', 'Title', 'Genres']
	movies = pd.read_table('movielens/ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python')
	titles = movies['Title'].str.split('(').str[0].str.strip()
	years = movies['Title'].str.split('(').str[1].str.replace(')', '').str.strip()
	movies['Title'] = titles
	movies['Year'] = years.apply(pd.to_numeric, errors='ignore')
	movies['Genres'] = movies['Genres'].str.split('|')
	result = []
	for index, row in movies.iterrows():
		if row['Year'] == year and genre in row['Genres']:
			result.append(row.tolist())

	return result
