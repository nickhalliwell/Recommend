#!/usr/bin/env python3

import requests
import os
import time
import gzip
import pandas as pd
from bs4 import BeautifulSoup
import time
import helper

#os.chdir('/Users/nhalliwe/Desktop/Amazon-data')
#pd.set_option('display.max_colwidth', -1)

def parse(path):
    g = gzip.open(path, 'rb')
    for i, l in enumerate(g):

        #if i == 300:
           #break
        yield eval(l)
        
def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

begin = time.time()
#meta = getDF('Amazon-data/meta_books_0_199.csv.gz')
meta = pd.read_csv('meta_books_0_199.csv.gz', compression='gzip')
end = time.time()

print(meta.head())