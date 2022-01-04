# -*- coding: utf-8 -*-
import os
import torch
import gudhi
import anndata
import numpy as np
import scanpy as sc
import squidpy as sq
import pandas as pd
import networkx as nx
from scipy.sparse import save_npz, load_npz
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


def load_DLPFC_data(args, sample_name, v2=True):
    if v2 and sample_name != '151675':
        file_fold = f'{args.dataset_dir}/DLPFC_v2/{sample_name}'
        adata = sc.read_10x_mtx(file_fold)
        adata.obsm['spatial'] = pd.read_csv(f"{file_fold}/spatial_coords.csv").values.astype(float)
    else:
        file_fold = f'{args.dataset_dir}/DLPFC/{sample_name}'
        adata = load_ST_file(file_fold=file_fold)
    return adata

def load_slideseqv2_data():
    adata = sq.datasets.slideseqv2()
    return adata

def load_chicken_data(args, sample_name):
    file_fold = f'{args.dataset_dir}/Visium/Chicken_Dev/ST/{sample_name}'
    adata = load_ST_file(file_fold=file_fold)
    return adata

def load_breast_cancer_data(args, sample_name):
    base_dir = f"{args.dataset_dir}/Visium/Breast_Cancer"
    count_fp = f'{base_dir}/ST-cnts/{sample_name}.tsv'
    adata = sc.read_csv(count_fp, delimiter='\t', first_column_names=True)
    coord_fp = f'{base_dir}/ST-spotfiles/{sample_name}_selection.tsv'
    coord_df = pd.read_csv(coord_fp, delimiter='\t')
    spots_idx_dicts = {f"{item[0]}x{item[1]}" : idx for idx, item in enumerate(coord_df[["x", "y"]].values.astype(int))}
    spots_selected = np.array([sid for sid, spot in enumerate(list(adata.obs_names)) if spot in spots_idx_dicts]).astype(int)
    adata = adata[spots_selected, :]
    coord_df = coord_df.iloc[spots_selected, :]
    spots_idx_dicts = {f"{item[0]}x{item[1]}": idx for idx, item in enumerate(coord_df[["x", "y"]].values.astype(int))}
    coords = coord_df[["pixel_x", "pixel_y"]].values
    adata.obsm["spatial"] = np.array([coords[spots_idx_dicts[spot]] for spot in adata.obs_names])
    return adata, spots_idx_dicts

def load_preprocessed_data(args, dataset, sample_name):
    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}/preprocessed'
    mkdir(data_root)
    adata = anndata.read_h5ad(f'{data_root}/adata.h5ad')
    spatial_graph = load_npz(f'{data_root}/spatial_graph.npz')
    print(f"Readed Preprocessed Data of {dataset}!")
    return adata, spatial_graph

def save_preprocessed_data(args, dataset, sample_name, adata, spatial_graph):
    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}/preprocessed'
    mkdir(data_root)
    adata.write(f'{data_root}/adata.h5ad')
    save_npz(f'{data_root}/spatial_graph.npz', spatial_graph)
    print(f"Saved Preprocessed Data of {dataset}!")

def preprocessing_data(args, adata, n_top_genes=None):
    sc.pp.filter_genes(adata, min_counts=1)  # only consider genes with more than 1 count
    sc.pp.normalize_per_cell(adata, key_n_counts='n_counts_all', min_counts=0)  # normalize with total UMI count per cell
    sc.pp.filter_genes_dispersion(adata, flavor='cell_ranger',log=False, subset=True, n_top_genes=n_top_genes)
    sc.pp.normalize_per_cell(adata, min_counts=0)  # renormalize after filtering
    sc.pp.log1p(adata)  # log transform: adata.X = log(adata.X + 1)
    sc.pp.pca(adata)  # log transform: adata.X = log(adata.X + 1)
    coords = adata.obsm['spatial']
    cut = estimate_cutoff_knn(coords, k=args.n_neighbors_for_knn_graph)
    spatial_graph = graph_alpha(coords, cut=cut, n_layer=args.alpha_n_layer)
    print('adata after filtered: (' + str(adata.shape[0]) + ', ' + str(adata.shape[1]) + ')')
    return adata, spatial_graph

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

def get_target_fp(args, dataset, sample_name):
    sp_suffix = "_SP" if args.spatial else ""
    method_dir = f"{args.arch}{sp_suffix}"
    return f"{dataset}/{sample_name}/{method_dir}"

def save_features(args, reduced_reprs, dataset, sample_name):
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    mkdir(output_dir)
    feature_fp = os.path.join(output_dir, f"features.tsv")
    np.savetxt(feature_fp, reduced_reprs[:, :], delimiter="\t")
    print(f"features saved successful at {feature_fp}")
