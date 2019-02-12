from config import *
import os
import numpy as np
from utils import get_result


def merge_folds(name,output_name):
    te_files = [output_dir+name+'fold'+str(i)+'.npy' for i in range(10)]
    pred = 0
    for file in te_files:
        pred += np.load(file)
    pred /= 10
    get_result(pred, output_name + '.txt')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()
name = args.name

import time
import datetime

print('wait fold parallel...')
time.sleep(3600*4)
for i in range(10):
    file = output_dir+name+'fold'+str(i)+'.npy'
    while os.path.exists(file)==False:
        time.sleep(300)

merge_folds(name,output_dir+'submit_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))