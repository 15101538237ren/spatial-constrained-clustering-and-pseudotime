# -*- coding: utf-8 -*-
import os
from python_codes.util.config import args
from python_codes.train.train import train
from python_codes.train.clustering import clustering
from python_codes.train.pseudotime import pseudotime
from python_codes.util.util import load_slideseqv2_data, preprocessing_data, save_preprocessed_data, load_preprocessed_data, save_features

dataset = "slideseq_v2"
clustering_method = "leiden"
sample_list = ['slideseq_v2']

for sample_idx, sample_name in enumerate(sample_list):
    print(f'===== Data {sample_idx + 1} : {sample_name}')
    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}/preprocessed'
    if os.path.exists(data_root):
        adata_filtered, spatial_graph = load_preprocessed_data(args, dataset, sample_name)
    else:
        adata = load_slideseqv2_data()
        adata_filtered, spatial_graph = preprocessing_data(args, adata)
        save_preprocessed_data(args, dataset, sample_name, adata, spatial_graph)
    for spatial in [False, True]:
        args.spatial = spatial
        embedding = train(args, adata_filtered, spatial_graph)
        save_features(args, embedding, dataset, sample_name)
        # clustering(args, dataset, sample_name, clustering_method)
        # pseudotime(args, dataset, sample_name, root_cell_type="WM", cell_types=anno_clusters)