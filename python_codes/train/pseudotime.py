# -*- coding: utf-8 -*-
import os
import pandas as pd
import scanpy as sc
import numpy as np
from scipy.spatial import distance_matrix
from python_codes.util.util import get_target_fp

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def pseudotime(args, dataset, sample_name, n_neighbors=20, resolution=1.0):
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    feature_fp = os.path.join(output_dir, "features.tsv")
    pseudotime_fp = os.path.join(output_dir, "pseudotime.tsv")
    if os.path.exists(feature_fp):
        adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=resolution)
        sc.tl.paga(adata)
        if adata.shape[0] < 5000:
            sub_adata_x = adata.X
        else:
            indices = np.arange(adata.shape[0])
            selected_ind = np.random.choice(indices, 5000, False)
            sub_adata_x = adata.X[selected_ind, :]
        sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
        adata.uns['iroot'] = np.argmax(sum_dists)
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata)
        pseudotimes = adata.obs['dpt_pseudotime'].to_numpy()
        np.savetxt(pseudotime_fp, pseudotimes, fmt='%.5f', header='', footer='', comments='')
        print("Saved %s succesful!" % pseudotime_fp)