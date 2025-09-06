
import requests
from bs4 import BeautifulSoup

class DouBan:
    def __init__(self):
        self.url = 'https://movie.douban.com/top250'
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                        (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        self.start_nums = []
        for i in range(0, 251, 25):
            self.start_nums.append(i)
        # print(self.start_nums)

    def get_top250_movie_names(self):
        top250_movie_names = []

        for start_num in self.start_nums:
            params = {'start': start_num, 'filter': ''}
            r = requests.get(self.url, headers=self.headers, params=params)
            html_doc = r.text
            soup = BeautifulSoup(html_doc, 'html.parser')
            # print(soup.prettify())
            # movie_names = soup.select('div.hd > a > span:nth-child(1)')
            movie_names = soup.select('#content > div > div.article > ol > li > div > div.info > div.hd > a > span:nth-child(1)')
            batch = [movie_name.text for movie_name in movie_names]
            top250_movie_names.extend(batch)

        print(len(top250_movie_names))
        print(top250_movie_names)










if __name__ == '__main__':
    douBan = DouBan()
    douBan.get_top250_movie_names()


















