#!/bin/bash

#load modulels for cuda and cudnn
module load cuda/9.1
module load cudnn/7.0-cuda-9.1
#get in the virtualenv (remember to pip install stuff you need before)
source ~/virtualenv/bin/activate

#launch your script
scrapy crawl book_spyder -o bookmooch_2000000_end.csv -t csv
#scrapy crawl book_spyder -o book_meta_0_20000.csv -t csv