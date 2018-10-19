
def process_user_data():
	print('processing users data....')
	# read users data
	users_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
	users = pd.read_table('movielens/ml-1m/users.dat', sep='::', header=None, names=users_title, engine='python')
	users = users.filter(regex='UserID|Gender|Age|OccupationID')
	users_original = users.values
	gender_map = {'F': 0, 'M': 1}
	users['Gender'] = users['Gender'].map(gender_map)
	age_map = {val: ii for ii, val in enumerate(sorted(list(set(users['Age']))))}
	users['Age'] = users['Age'].map(age_map)
	return users, users_original


def process_movie_data():
	print('processing movies data....')
	# read movies data
	movies_title = ['MovieID', 'Title', 'Genres']
	movies = pd.read_table('movielens/ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python')
	movies_original = movies.values

	movies['Title'] = movies['Title'].str.split('(').str[0].str.strip()
	title_to_id_df = movies[['Title', 'MovieID']]

	title_set = set()
	for val in movies['Title'].str.split():
		title_set.update(val)

	title_set.add('PADDING')
	title2int = {val: ii for ii, val in enumerate(title_set)}

	title_length = 15
	title_map = {val: [title2int[row] for row in val.split()] for val in set(movies['Title'])}

	for key in title_map.keys():
		padding_length = title_length - len(title_map[key])
		padding = [title2int['PADDING']] * padding_length
		title_map[key].extend(padding)

	movies['Title'] = movies['Title'].map(title_map)

	genres_set = set()
	for val in movies['Genres'].str.split('|'):
		genres_set.update(val)

	genres_set.add('PADDING')
	genres2int = {val: ii for ii, val in enumerate(genres_set)}

	genres_map = {val: [genres2int[row] for row in val.split('|')] for val in set(movies['Genres'])}

	for key in genres_map:
		for cnt in range(max(genres2int.values()) - len(genres_map[key])):
			genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['PADDING'])

	movies['Genres'] = movies['Genres'].map(genres_map)
	return movies, movies_original, genres2int, title_set, title_length, title_to_id_df
