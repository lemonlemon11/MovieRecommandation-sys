from flask import Flask
from flask import request
from flask import Flask, render_template, request, jsonify,url_for
from pymongo import MongoClient
import uuid
import requests

from flask_restplus import Resource, Api, fields, reqparse
import recommendation
import json
import get_movie_info
import utils
import random as r
app = Flask(__name__)

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

people_list=[]

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
        movie_info_dict['id'] = movie[0]
        movie_info_dict['movie_name'] = movie[1]
        movie_info_dict['type'] = movie[2]
        movie_info_dict['img_url']='https://images.summitmedia-digital.com/spotph/images/2017/12/14/MoviePosters2017_2.jpg'
        processed_movies.append(movie_info_dict)
    return processed_movies


@app.route('/movie_recommendation/other_favorite')  # 返回其他看过该电影的人也爱看的电影
def recommand_other_favorite():
    movie_id = int(request.args['query'])
    print('movie_id', movie_id)
    data_url = 'http://127.0.0.1:9999/movie_recommendation/other_favorite?query={}'.format(movie_id)
    resp = requests.get(url=data_url)

    movie_you_watched, recom_other_favorite_movies, users_info = recommendation.recommend_other_favorite_movie(movie_id)
    users_info = process_users_info(users_info)
    recom_other_favorite_movies = resp.json()['recom_other_favorite_movies']

    people_dict=users_info
    global  people_list
    people_list=people_dict
    return jsonify(recom_other_favorite_movies)


@app.route('/movie_recommendation/same_type')    # 根据看过的电影返回同类型推荐电影
def recommand_same_type():
    movie_id=int(request.args['query'])
    print('movie_id',movie_id)
    data_url='http://127.0.0.1:9999/movie_recommendation/same_type?query={}'.format(movie_id)
    resp = requests.get(url=data_url)
    recom_same_type_movies=resp.json()['recom_same_type_movies']
    return jsonify(recom_same_type_movies)


@app.route('/movie_recommendation/get_movie_name_by_year_type')
def get_movie_name():
    year=int(request.args['year'])
    movie_type=request.args['movie_type']
    print(year,movie_type)
    movie_info=utils.get_movie_list_by_year_type(year,movie_type)
    info_list=[]
    for data in movie_info:
        d={}
        d['id']=data[0]
        d['movie_name']=data[1]
        info_list.append(d)
    return jsonify(info_list)



@app.route('/movie_recommendation/get_movie_dict')
def get_movie_dict():
    return jsonify(utils.movie_type_to_dict())


@app.route('/index.html',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/welcome.html',methods=['GET','POST'])
def welcome():
    return render_template('welcome.html')

@app.route('/',methods=['GET','POST'])
def login():
    return render_template('login.html')

@app.route('/get-id-same-type.html',methods=['GET','POST'])
def get_id_same_type():
    return render_template('get-id-same-type.html')

@app.route('/search-same-type',methods=['GET','POST'])
def search_same_type():
    movieid = request.args['movie_id']
    print(movieid)
    return render_template('data-table.html',movieid=movieid)

@app.route('/get-other-favorite.html',methods=['GET','POST'])
def get_other_favorite():
    return render_template('get-other-favorite.html')

@app.route('/search-other-favorite',methods=['GET','POST'])
def search_other_favorite():
    movieid = request.args['movie_id']
    return render_template('data-table2.html',movieid=movieid)

@app.route('/people-list.html',methods=['GET','POST'])
def people_list():
    return render_template('people-list.html')

@app.route('/movie_recommendation/get_favourite_people',methods=['GET','POST'])
def get_favourite_people():
    # print(people_list)
    d={}
    people_info=[]
    for data in people_list:
        people_info.append(data)
    d['data'] = people_info
    d['draw'] = 1
    d['recordsTotal'] = len(people_list)
    d['recordsFiltered'] = len(people_list)
    return jsonify(d)



@app.route('/recommend-user-movie.html',methods=['GET','POST'])
def recommend_user_movie():
    return render_template('recommend-user-movie.html')


@app.route('/movie_recommendation/get_recommend_user_list',methods=['GET','POST'])
def get_recommend_user_list():
    names_list=['Json','Tracy','Bill','Tom','Mary','Ted','Tim','Zed','Neo','Steven']
    r.shuffle(names_list)
    user_dict={}
    user_list=[]
    for i in range(0,10):
        d={}
        user_id=r.randrange(0,1000)
        d['id']=user_id
        d['name']=names_list[i]
        user_list.append(d)
    user_dict['data']=user_list
    return jsonify(user_dict)

@app.route('/recommend-user-list.html',methods=['GET','POST'])
def recommend_user_list():
    return render_template('recommend-user-list.html')

@app.route('/recommend-user-movie-list',methods=['GET','POST'])
def recommend_user_movie_list():
    user_id = request.args['user_id']
    return render_template('data-table3.html',user_id=user_id)

@app.route('/movie_recommendation/recommend_user_movie_list',methods=['GET','POST'])
def movie_recommendation_recommend_user_movie_list():
    user_id = request.args['query']
    data_url = 'http://127.0.0.1:9999/movie_recommendation/users?query={}'.format(user_id)
    resp = requests.get(url=data_url)
    recom_movies = resp.json()['recom_movies']
    return jsonify(recom_movies)



@app.route('/movie_recommendation/user_register',methods=['GET','POST'])
def user_register():
    return render_template('register.html')

@app.route('/movie_recommendation/user_login',methods=['GET','POST'])
def user_login():
    return render_template('login.html')

@app.route('/movie_recommendation/user_login_check',methods=['GET','POST'])
def user_login_check():
    account = request.args['account']
    password=request.args['password']

    conn=get_connection()
    users=conn.find()
    flag=False
    for user in users:
        if user['account']==account and user['password']==password:
            flag=True
            break
    if flag:
        d1={}
        d1['error']=200
        return str(200)
    else:
        d2 = {}
        d2['error'] = 400
        return str(400)

@app.route('/movie_recommendation/save_user_to_database',methods=['GET','POST'])
def save_user():
    account = request.args['account']
    password=request.args['password']
    id=str(uuid.uuid1())
    user_info={}
    user_info['id']=id
    user_info['account']=account
    user_info['password']=password
    print(user_info)
    print(json.dumps(user_info))
    conn=get_connection()
    conn.insert(user_info)
    return render_template('login.html')


def get_connection():
    db_name = 'comp9321'
    collection = 'user'
    client = MongoClient("mongodb://root:ltf9495!@ds245772.mlab.com:45772/{db}".format(db=db_name))
    db = client[db_name]
    c = db[collection]
    return c
if __name__ == '__main__':

    app.run(port=9999)

