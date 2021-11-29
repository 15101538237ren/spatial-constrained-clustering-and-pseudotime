# -*- coding: utf-8 -*-
from python_codes.util.config import args
from python_codes.train.train import train
from python_codes.train.clustering import clustering
from python_codes.train.pseudotime import pseudotime
from python_codes.visualize.chicken import *
from python_codes.util.util import load_chicken_data, preprocessing_data, save_features
dataset = "chicken"
clustering_method = "leiden"
resolution = 1.0
n_neighbors = 6
sample_list = ['D4', 'D7', 'D10', 'D14']

for sample_name in sample_list:
    anno_clusters = get_annotations_chicken(args, sample_name)
    for spatial in [False, True]:
        args.spatial = spatial
        adata = load_chicken_data(args, sample_name)
        adata_filtered, expr, genes, cells, spatial_graph, spatial_dists = preprocessing_data(args, adata)
        embedding = train(args, expr, spatial_graph, spatial_dists)
        save_features(args, embedding, dataset, sample_name)
        clustering(args, dataset, sample_name, clustering_method, n_neighbors=n_neighbors, resolution=resolution)
        pseudotime(args, dataset, sample_name, root_cell_type="Epi-epithelial cells", cell_types=anno_clusters, n_neighbors=n_neighbors, resolution=resolution)
    plot_clustering(args, sample_name, clustering_method)
    plot_pseudotime(args, sample_name)