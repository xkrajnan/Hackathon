from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import re
import time

def get_html(url):
    try:
        with closing(get(url, stream=True)) as resp:
            ctype = resp.headers['Content-Type'].lower()

            if resp.status_code == 200 and ctype is not None and ctype.find('html') > -1:
                return resp.content
            else:
                return None

    except RequestException as e:
        return None

def get_links(raw_html):
    root = BeautifulSoup(raw_html, 'html.parser')
    mainsite = root.findAll('div', {'class': 'site-main'})[0]
    return [article.find('a')['href'] for article in mainsite.findAll('article')]

def get_article_text(raw_html):
    root = BeautifulSoup(raw_html, 'html.parser')
    article = root.find('article').find('div', {'class': 'entry-content'})
    texts = article.findAll('p', {'style': 'text-align: justify;'})
    parsed = []

    for text in texts:
        try:
            no_escapes = re.sub(r"([#\\?])(\w+)\b", ' ', text.find(text=True))
            parsed.append(' '.join(no_escapes.split()))
        except:
            pass

    return parsed


if __name__ == '__main__':
    for page_num in range(21, 40):
        texts = []
        time.sleep(2)
        print('index:', page_num)
        
        index_html = get_html('https://www.amodelrecommends.com/category/beauty/page/' + str(page_num) + '/')
        links = get_links(index_html)

        for link in links:
            print(link)
            time.sleep(3)
            article_html = get_html(link)
            texts = texts + get_article_text(article_html)
            

        with open('parsed_amodelrecommends' + str(page_num) + '.txt', 'w') as outfile:
            for text in texts:
                outfile.write(text)
                outfile.write('\n')
