#!/bin/bash

#load modulels for cuda and cudnn
module load cuda/9.1
module load cudnn/7.0-cuda-9.1
#get in the virtualenv (remember to pip install stuff you need before)
source ~/virtualenv/bin/activate

#launch your script
./bookmooch.py
