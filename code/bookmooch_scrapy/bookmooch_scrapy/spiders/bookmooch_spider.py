#!/usr/bin/env python3

import scrapy
import pandas as pd
import gzip
import time

def parse(path):
    
    g = gzip.open(path, 'rb')
    
    for i, l in enumerate(g):
            
        yield eval(l)
        
def getDF(path):
    
    i = 0
    
    df = {}
    
    for d in parse(path):
        
        df[i] = d
        
        i += 1
        
    return pd.DataFrame.from_dict(df, orient='index')

meta = getDF('/home/nhalliwe/Recommend/Amazon-data/meta_Books.json.gz')
#meta = getDF('/Users/nhalliwe/Desktop/Amazon-data/meta_Books.json.gz')


class BookSpyderSpider(scrapy.Spider):

    name = 'book_spyder'

    #allowed_domains = ['bookmooch.com']

    def start_requests(self):

    	urls = meta['asin'].values

    	for prod_id in urls:

            url = 'http://api.bookmooch.com/api/asin?asins=' + str(prod_id)


            yield scrapy.Request(url = url, callback=self.parse, meta = {'prod_id': str(prod_id)})

    def parse(self, response):
        
         book_id = response.meta['prod_id']

         author = response.xpath('/asins/asin/Author/text()').extract_first()

         publisher = response.xpath('/asins/asin/Publisher/text()').extract_first()

         genres = response.xpath('/asins/asin/Topics/item/text()').extract()

         yield {"asin":book_id, "Author": author, "Publisher":publisher, "Genre":genres}
