# -*- coding: utf-8 -*-
import os
import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.cluster import KMeans

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def res_search_fixed_clus(adata, n_cluster, increment=0.02):
    for res in sorted(list(np.arange(0.02, 2, increment)), reverse=False):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        print("Try resolution %3f found %d clusters: target %d" % (res, count_unique_leiden, n_cluster))
        if count_unique_leiden == n_cluster:
            print("Found resolution: %.3f" % res)
            return res
        elif count_unique_leiden > n_cluster:
            print("Found resolution: %.3f" % (res - increment))
            return res - increment

def leiden(adata, ncluster=7):
    resolution = res_search_fixed_clus(adata, ncluster)
    sc.tl.leiden(adata, resolution=float(resolution))
    labels = adata.obs["leiden"].cat.codes
    return labels

def louvain(adata, ncluster=7):
    resolution = res_search_fixed_clus(adata, ncluster)
    sc.tl.louvain(adata, resolution=float(resolution))
    labels = adata.obs["louvain"].cat.codes
    return labels

def kmeans(adata, ncluster=7):
    kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(adata.X)
    labels = kmeans.labels_
    return labels

def clustering(args, dataset, sample_name, method = "leiden", n_neighbors=50):
    feature_dir = f'{args.feature_dir}/{dataset}'
    cluster_dir = f'{args.cluster_dir}/{dataset}'
    mkdir(cluster_dir)
    sp_suffix = "_sp" if args.spatial else ""
    feature_fp = os.path.join(feature_dir, f"{sample_name}{sp_suffix}.tsv")
    cluster_fp = os.path.join(cluster_dir, f"{sample_name}{sp_suffix}_{method}.tsv")

    if os.path.exists(feature_fp):
        adata_feat = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
        sc.pp.neighbors(adata_feat, n_neighbors=n_neighbors, use_rep='X')
        labels = leiden(adata_feat)
        np.savetxt(cluster_fp, labels, fmt='%d', header='', footer='', comments='')
        print("Saved %s succesful!" % cluster_fp)
    else:
        print(f"Error in clustering, the feature fp: {feature_fp} not exist")