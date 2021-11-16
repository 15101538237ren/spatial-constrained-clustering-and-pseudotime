# -*- coding: utf-8 -*-
from python_codes.util.config import args
from python_codes.util.util import load_DLPFC_data, preprocessing_data, save_features
from python_codes.models.model_hub import get_model
from python_codes.train.clustering import clustering
from python_codes.train.train import train

dataset = "DLPFC"
sample_list = ['151507']
# , '151508', '151509', '151510',
#              '151669', '151670', '151671', '151672',
#              '151673', '151674', '151675', '151676'

for sample_idx, sample_name in enumerate(sample_list):
    print(f'===== Project {sample_idx + 1} : {sample_name}')
    adata = load_DLPFC_data(args, sample_name)
    expr, genes, cells, spatial_graph, spatial_dists = preprocessing_data(args, adata)
    model = get_model(args, len(genes))
    embedding = train(args, model, expr, spatial_graph, spatial_dists)
    save_features(args, embedding, dataset, sample_name)
    clustering(args, dataset, sample_name)