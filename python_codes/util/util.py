# -*- coding: utf-8 -*-
import os
import torch
import gudhi
import numpy as np
import scanpy as sc
import pandas as pd
import networkx as nx
from scipy.spatial import distance
from sklearn.neighbors import kneighbors_graph

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_ST_file(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True, file_Adj=None):
    adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata_h5.var_names_make_unique()

    if load_images is False:
        if file_Adj is None:
            file_Adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        adata_h5.obs.drop(columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True)
        print('adata: (' + str(adata_h5.shape[0]) + ', ' + str(adata_h5.shape[1]) + ')')
    return adata_h5


def load_DLPFC_data(args, sample_name):
    data_root = f'{args.dataset_dir}/DLPFC'
    file_fold = f'{data_root}/{sample_name}'
    adata_h5 = load_ST_file(file_fold=file_fold)
    return adata_h5

def preprocessing_data(args, adata):
    sc.pp.filter_genes(adata, min_counts=1)  # only consider genes with more than 1 count
    sc.pp.normalize_per_cell(adata, key_n_counts='n_counts_all', min_counts=0)  # normalize with total UMI count per cell
    filter_result = sc.pp.filter_genes_dispersion(adata.X, flavor='cell_ranger',log=False)
    adata_X = adata[:, filter_result.gene_subset]  # subset the genes

    sc.pp.normalize_per_cell(adata_X, min_counts=0)  # renormalize after filtering
    sc.pp.log1p(adata_X)  # log transform: adata.X = log(adata.X + 1)

    print('adata after filtered: (' + str(adata_X.shape[0]) + ', ' + str(adata_X.shape[1]) + ')')

    genes = list(adata_X.var_names)
    cells = list(adata_X.obs_names)
    coords = adata_X.obsm['spatial']
    expr = adata_X.X.todense() if type(adata_X.X).__module__ != np.__name__ else adata_X.X
    expr = torch.tensor(expr).float()
    cut = estimate_cutoff_knn(coords, k=args.n_neighbors_for_knn_graph)
    spatial_graph = graph_alpha(coords, cut=cut, n_layer=args.alpha_n_layer)
    spatial_dists = distance.cdist(coords, coords, 'euclidean')
    spatial_dists = torch.tensor((spatial_dists / np.max(spatial_dists))).float()
    return expr, genes, cells, spatial_graph, spatial_dists

def estimate_cutoff_knn(pts, k=10):
    A_knn = kneighbors_graph(pts, n_neighbors=k, mode='distance')
    est_cut = A_knn.sum() / float(A_knn.count_nonzero())
    return est_cut

def graph_alpha(pts, n_layer=1, cut=np.inf):
    # Get a graph from alpha shape
    pts_list = pts.tolist()
    n_node = len(pts_list)
    alpha_complex = gudhi.AlphaComplex(points=pts_list)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=cut ** 2)
    skeleton = simplex_tree.get_skeleton(1)
    initial_graph = nx.Graph()
    initial_graph.add_nodes_from([i for i in range(n_node)])
    for s in skeleton:
        if len(s[0]) == 2:
            initial_graph.add_edge(s[0][0], s[0][1])
    # Extend the graph for the specified layers
    extended_graph = nx.Graph()
    extended_graph.add_nodes_from(initial_graph)
    extended_graph.add_edges_from(initial_graph.edges)
    if n_layer == 2:
        for i in range(n_node):
            for j in initial_graph.neighbors(i):
                for k in initial_graph.neighbors(j):
                    extended_graph.add_edge(i, k)
    elif n_layer == 3:
        for i in range(n_node):
            for j in initial_graph.neighbors(i):
                for k in initial_graph.neighbors(j):
                    for l in initial_graph.neighbors(k):
                        extended_graph.add_edge(i, l)
    if n_layer >= 4:
        print("Setting n_layer to greater than 3 may results in too large neighborhoods")

        # Remove self edges
    for i in range(n_node):
        try:
            extended_graph.remove_edge(i, i)
        except:
            pass

    return nx.to_scipy_sparse_matrix(extended_graph, format='csr')

def save_features(args, reduced_reprs, dataset, sample_name):
    feature_dir = f'{args.feature_dir}/{dataset}'
    mkdir(feature_dir)
    sp_suffix = "_sp" if args.spatial else ""
    feature_fp = os.path.join(feature_dir, f"{sample_name}{sp_suffix}.tsv")
    np.savetxt(feature_fp, reduced_reprs[:, :], delimiter="\t")
    print(f"features saved successful at {feature_fp}")
