import requests
from bs4 import BeautifulSoup


class movie:
    def __init__(self, name):
        self.name = name

    def get_movie_info(self):
        search_url = 'https://www.imdb.com/find?ref_=nv_sr_fn&q=' + self.name + '&s=all'
        search_html = requests.get(search_url).text
        search_page = BeautifulSoup(search_html, 'lxml')

        match = search_page.find('td', class_="result_text")

        if match:
            return self.get_detail(match.a['href'])
        return None

    #
    def get_detail(self, link):
        url = 'https://www.imdb.com' + link
        result = dict()

        res = requests.get((url)).text
        page = BeautifulSoup(res, 'lxml')

        poster = page.find('div', class_='poster')
        poster_url = poster.a.img['src']

        summary = page.find('div', class_='summary_text').text.strip()

        rate = page.find('div', class_='ratingValue').strong['title']
        # print('rate: ',rate)

        people = page.find_all('div', class_='credit_summary_item')
        # print(people)

        info = page.find('div', class_='subtext')
        stars = []
        director = ''

        # director = page.find('a', href = re.compile("/name[...]dr"))
        # print(director)

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
        # print('directors: ', director)
        # print('stars: ', stars)

        time = info.time.text.strip()
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

        return result


def get_movie_info(movie_list):
    res = []
    for title in movie_list:
        m_info = movie(title).get_movie_info()
        res.append(m_info)
    return res
