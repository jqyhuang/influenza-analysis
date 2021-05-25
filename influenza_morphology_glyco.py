#!/usr/bin/env python

import os
import sys
import argparse
import open3d as o3d
import numpy as np
import pandas as pd
from sklearn.cluster import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('coordinates',
                        type=str,
                        help='Coordinates to be transformed into point clouds')
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
    parser.add_argument('-n',
                        '--nvirion',
                        type=int,
                        dest='nvirion',
                        default=None,
                        help='Number of virions')

    return parser.parse_args()


def inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(inlier_cloud)
    vis.add_geometry(outlier_cloud)
    vis.run()
    vis.capture_screen_image(os.path.join(datadir,"{}_inlier_outlier.png".format(prefix)))
    vis.destroy_window()

args=parse_args()
datadir = args.datadir
coordinates = args.coordinates
prefix = args.prefix
nvirion = args.nvirion

pcd = o3d.io.read_point_cloud(coordinates, format='xyz')
pcd.paint_uniform_color([0.8, 0.8, 0.8])
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=200,std_ratio=0.1)
inlier_outlier(pcd, ind)
cl_array=np.asarray(cl.points)
df=pd.DataFrame(np.asarray(cl.points), columns=['x','y','z'])
df=df.set_index('x')
df.to_csv(os.path.join(datadir, '{}_glyco.csv'.format(prefix)))

y_pred = KMeans(n_clusters=nvirion).fit(cl_array)
labels=pd.DataFrame(y_pred.labels_)
labels.to_csv(os.path.join(datadir, '{}_labels.csv'.format(prefix)))
labels=pd.read_csv(os.path.join(datadir, '{}_labels.csv'.format(prefix)))

for i in range(nvirion-1):
    result=np.where(labels==i)
    len(result)
    cloud_singlevirion=cl.select_by_index(result[0])
    cloud_singlevirion.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1000, max_nn=100))
    cloud_singlevirion.orient_normals_consistent_tangent_plane(100)
    print('run Poisson surface reconstruction')
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud_singlevirion, depth=9)
    print('{}_{}'.format(i,mesh))
        #o3d.visualization.draw_geometries([mesh])
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(datadir, '{}_{}_mesh.stl'.format(prefix,i)), mesh)
    cloud_array=np.asarray(cloud_singlevirion.points)
    df=pd.DataFrame((cloud_array), columns=['x','y','z'])
    df=df.set_index('x')
    df.to_csv(os.path.join(datadir, '{}_{}_glyco.csv'.format(prefix,i)))



