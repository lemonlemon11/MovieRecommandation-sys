import pandas as pd
def movie_type_to_dict():
    l=[]
    type_list=['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
    for idx in range(0,len(type_list)):
        type_dict = {}

        type_dict['id']=idx
        type_dict['movie_type']=type_list[idx]
        l.append(type_dict)
    return l


def get_movie_list_by_year_type(year, genre):
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

