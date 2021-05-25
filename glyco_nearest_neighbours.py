#!/usr/bin/env python


import future
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('glycos',
                        type=str,
                        help='Coordinates from previously cleaned up point clouds')
    parser.add_argument('--prefix',
                        type=str,
                        dest='prefix',
                        default='test',
                        help='Outfile prefix')
    parser.add_argument('-d',
                        '--datadir',
                        type=str,
                        dest='datadir',
                        default=os.getcwd(),
                        help='Destination of saved files')
    parser.add_argument('-p',
                        '--pixel',
                        type=float,
                        dest='pixel',
                        default=None,
                        help='pixel size in nm')

    return parser.parse_args()

args=parse_args()
datadir = args.datadir
glycos = args.glycos
prefix = args.prefix
pixel = args.pixel

vectors = []
with open(glycos) as csvfile:
    reader = csv.reader(csvfile,quoting=csv.QUOTE_ALL)
    for row in reader: 
        vectors.append(row)
K_th = 1
vectorarray = np.array(vectors)
K_corr = K_th + 3
nbrs = NearestNeighbors(n_neighbors=K_corr, algorithm='brute').fit(vectorarray)
results, indices = nbrs.kneighbors(vectorarray)
dist = np.delete(results, np.s_[:K_th:], 1)
dist = np.multiply(dist, pixel)
print(dist[:10])
dist_df=pd.DataFrame(dist)
dist_df.to_csv(os.path.join(datadir,("{}_nbr.csv").format(prefix)))



