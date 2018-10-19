import requests
from bs4 import BeautifulSoup
import threading
import time
from copy import deepcopy

class movie:
    def __init__(self, name):
        self.name = name
        self.movie_name = deepcopy(name)
        self.res = []
        self.final_res = []
        t1 = threading.Thread(target=self.get_movie_info, name='t1')
        t1.start()
        t2 = threading.Thread(target=self.get_movie_info, name='t2')
        t2.start()
        t3 = threading.Thread(target=self.get_movie_info, name='t3')
        t3.start()
        t4 = threading.Thread(target=self.get_movie_info, name='t4')
        t4.start()
        t5 = threading.Thread(target=self.get_movie_info, name='t5')
        t5.start()

        while 1:
            if len(self.res) == 5:
                break

        self.re_order()

    def re_order(self):
        # print(len(self.res))
        # print(len(self.name))
        for i in range(5):
            for j in range(5):
                if self.movie_name[i] == self.res[j]['movie_name']:
                    self.final_res.append(self.res[j])



    def get_movie_info(self):
        name = self.name.pop()
        search_url = 'https://www.imdb.com/find?ref_=nv_sr_fn&q=' + name + '&s=all'
        search_html = requests.get(search_url).text
        search_page = BeautifulSoup(search_html, 'lxml')

        match = search_page.find('td', class_="result_text")

        if match:
            self.get_detail(match.a['href'], name)
            return
        return None

    #
    def get_detail(self, link, name):
        url = 'https://www.imdb.com' + link
        result = dict()
        res = requests.get((url)).text
        page = BeautifulSoup(res, 'lxml')

        poster = page.find('div', class_='poster')
        try:
            poster_url = poster.a.img['src']
        except:
            poster_url = None

        try:
            summary = page.find('div', class_='summary_text').text.strip()
        except:
            summary = None

        try:
            rate = page.find('div', class_='ratingValue').strong['title']
        except:
            rate = None
        # print('rate: ',rate)
        try:
            people = page.find_all('div', class_='credit_summary_item')
        except:
            people = None
        # print(people)
        try:
            info = page.find('div', class_='subtext')
            time = info.time.text.strip()
        except:
            time = None
        stars = []
        director = ''

        # director = page.find('a', href = re.compile("/name[...]dr"))
        # print(director)
        try:
            for e in people:
                # print(type(e))
                # print(e)
                # print()
                if e.h4.text == 'Director:':
                    director = e.a.text
                elif e.h4.text == 'Stars:':
                    for i in e.find_all('a'):
                        stars.append(i.text)
            stars = stars[0:-1]
        except:
            stars = None
            director = None
        # print('directors: ', director)
        # print('stars: ', stars)

        # genre = info.a[0:-1]
        # release_date = info.a[-1]

        # print('time: ', time)
        # print('genre: ', genre)
        # print('release date: ', release_date)

        result['poster'] = poster_url
        result['summary'] = summary
        result['time'] = time
        result['director'] = director
        result['stars'] = stars
        result['rate'] = rate
        result['movie_name'] = name
        self.res.append(result)


def get_movie_info(movie_list):
    res = movie(movie_list).final_res
    # print(res)
    return res

# movie_list = ['maze runner', 'the matrix', 'destroyer', 'The Shawshank Redemption', 'The Godfather']
#
# head = time.time()
# get_movie_info(movie_list)
# end = time.time()
# print()
