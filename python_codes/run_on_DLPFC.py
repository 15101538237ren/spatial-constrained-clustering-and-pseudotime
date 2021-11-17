# -*- coding: utf-8 -*-
from python_codes.util.config import args
# from python_codes.train.train import train
from python_codes.train.clustering import clustering
from python_codes.train.pseudotime import pseudotime
from python_codes.visualize.dlpfc import plot_clustering, plot_pseudotime, get_annotations
# from python_codes.util.util import load_DLPFC_data, preprocessing_data, save_features

dataset = "DLPFC"
clustering_method = "kmeans"
sample_list = ['151507', '151508', '151509',
               '151510', '151671', '151669',
               '151670', '151672', '151673',
               '151674', '151675', '151676']

for sample_idx, sample_name in enumerate(sample_list):
    print(f'===== Project {sample_idx + 1} : {sample_name}')
    anno_clusters = get_annotations(args, sample_name)
    for spatial in [False, True]:
        args.spatial = spatial
        # adata = load_DLPFC_data(args, sample_name)
        # expr, genes, cells, spatial_graph, spatial_dists = preprocessing_data(args, adata)
        # embedding = train(args, expr, spatial_graph, spatial_dists)
        # save_features(args, embedding, dataset, sample_name)
        #clustering(args, dataset, sample_name, clustering_method)
        pseudotime(args, dataset, sample_name, root_cell_type="WM", cell_types=anno_clusters)
    #plot_clustering(args, sample_name, clustering_method)
    plot_pseudotime(args, sample_name)