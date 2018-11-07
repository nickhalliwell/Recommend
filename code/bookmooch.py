#!/usr/bin/env python3

import requests
import os
import time
import gzip
import pandas as pd
from bs4 import BeautifulSoup
import time

os.chdir('..')

def parse(path):
    g = gzip.open(path, 'rb')

    for i, l in enumerate(g):
        if i == 100000:
            break
        yield eval(l)
        
def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

begin = time.time()
meta = getDF('Amazon-data/meta_Books.json.gz')
end = time.time()

print('Data loaded in {}'.format(end-begin))

def call(start, stop):

    denied = []
    publishers = []
    authors = []
    genres = []
    ids = []

    for product_id in meta['asin'].loc[start:stop]:

        url = 'http://api.bookmooch.com/api/asin?asins=' + str(product_id)

        try:

            time.sleep(.1)

            text = requests.get(url).text

            soup = BeautifulSoup(text, 'xml')

            publisher = soup.find('Publisher').get_text(' ',strip=True)

            author = soup.find('Author').get_text(' ',strip=True)

            genre = soup.find('Topics').find('item').get_text(strip = True)#list(soup.find('Topics').find_all('item'))#[0].get_text(' ',strip=True)


            ids.append(product_id)

            publishers.append(publisher)

            authors.append(author)

            genres.append(genre)

        except:

            denied.append(product_id)

    out = pd.DataFrame({'asin' : ids, 'Publisher' : publishers, 'Authors' : authors, 'Genre': genres})

    return out, denied

start = 0
stop = 20000

out, denied = call(start, stop)

filename = 'Amazon-data/meta_books_{}_{}.csv'.format(start, stop-1)

out.to_csv(filename,index=False)


denied_path = 'Amazon-data/denied_{}_{}.txt'.format(start, stop-1)


with open(denied_path, 'w') as denied_out:

     denied_out.write(str(denied))