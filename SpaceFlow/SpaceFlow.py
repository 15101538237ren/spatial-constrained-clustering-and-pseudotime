# -*- coding: utf-8 -*-
import os
import torch
import random
import gudhi
import anndata
import numpy as np
import scanpy as sc
import networkx as nx
import torch.nn as nn
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from sklearn.neighbors import kneighbors_graph
from SpaceFlow.util import sparse_mx_to_torch_edge_list, corruption

class SpaceFlow(object):
    """An object for analysis of spatial transcriptomics data.

    :param expr_data: count matrix of gene expression, 2D numpy array of size (n_cells, n_genes)
    :type expr_data: class:`numpy.ndarray`
    :param spatial_locs: spatial locations of cells (or spots) match to rows of the count matrix, 1D numpy array of size (n_locations,)
    :type spatial_locs: class:`numpy.ndarray`

    List of instance attributes:

    :ivar expr_data: count matrix of gene expression, 2D numpy array of size (n_cells, n_genes) ``__init__``
    :vartype expr_data: class:`numpy.ndarray`
    :ivar spatial_locs: spatial locations of cells (or spots) match to rows of the count matrix, 1D numpy array of size (n_locations,) ``__init__``
    :vartype spatial_locs: class:`numpy.ndarray`

    """

    def __init__(self, expr_data, spatial_locs):
        """
        Inputs
        ------
        expr_data : count matrix of gene expression, 2D numpy array of size (n_cells, n_genes)
        spatial_locs : spatial locations of cells (or spots) match to rows of the count matrix, 1D numpy array of size (n_locations,)
        """
        self.adata = anndata.AnnData(expr_data.astype(float))
        self.adata.obsm['spatial'] = spatial_locs.astype(float)

    def preprocessing_data(self, adata, n_top_genes=None):
        """
        Preprocessing the spatial transcriptomics data
        Generates:  `self.adata_filtered`: (n_cells, n_locations) `numpy.ndarray`
                    `self.spatial_graph`: (n_cells, n_locations) `numpy.ndarray`
        :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
        :type adata: class:`anndata.annData`
        :param n_top_genes: the number of top highly variable genes
        :type n_top_genes: int, optional
        :return: a preprocessed annData object of the spatial transcriptomics data
        :rtype: class:`anndata.annData`
        :return: a geometry-aware spatial proximity graph of the spatial spots of cells
        :rtype: class:`scipy.sparse.csr_matrix`
        """

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
        sc.pp.pca(adata)
        spatial_locs = adata.obsm['spatial']
        spatial_graph = self.graph_alpha(spatial_locs)

        self.adata_preprocessed = adata
        self.spatial_graph = spatial_graph

        return self.adata_preprocessed, self.spatial_graph

    def graph_alpha(self, spatial_locs, n_neighbors_for_knn_graph=10):
        """
        Construct a geometry-aware spatial proximity graph of the spatial spots of cells by using alpha complex.
        :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
        :type adata: class:`anndata.annData`
        :param n_neighbors_for_knn_graph: the number of nearest neighbors for the knn graph, which is here as an estimation of the graph cut for building alpha complex graoh
        :type n_neighbors_for_knn_graph: int, optional, default: 10
        :return: a geometry-aware spatial proximity graph of the spatial spots of cells
        :rtype: class:`scipy.sparse.csr_matrix`
        """
        A_knn = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors_for_knn_graph, mode='distance')
        estimated_graph_cut = A_knn.sum() / float(A_knn.count_nonzero())
        spatial_locs_list = spatial_locs.tolist()
        n_node = len(spatial_locs_list)
        alpha_complex = gudhi.AlphaComplex(points=spatial_locs_list)
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=estimated_graph_cut ** 2)
        skeleton = simplex_tree.get_skeleton(1)
        initial_graph = nx.Graph()
        initial_graph.add_nodes_from([i for i in range(n_node)])
        for s in skeleton:
            if len(s[0]) == 2:
                initial_graph.add_edge(s[0][0], s[0][1])

        extended_graph = nx.Graph()
        extended_graph.add_nodes_from(initial_graph)
        extended_graph.add_edges_from(initial_graph.edges)

        # Remove self edges
        for i in range(n_node):
            try:
                extended_graph.remove_edge(i, i)
            except:
                pass

        return nx.to_scipy_sparse_matrix(extended_graph, format='csr')

    def train(self, adata_filtered, spatial_graph, embedding_save_filepath="./embedding.tsv", spatial_regularization_strength=0.1, z_dim=50, lr=0.001, epochs=1000, max_patience=50, min_stop=100, random_seed=42, gpu=0):

        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
        model = DeepGraphInfomax(
            hidden_channels=z_dim, encoder=GraphEncoder(adata_filtered.shape[1], z_dim),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption).to(device)

        expr = adata_filtered.X.todense() if type(adata_filtered.X).__module__ != np.__name__ else adata_filtered.X
        expr = torch.tensor(expr).float().to(device)

        edge_list = sparse_mx_to_torch_edge_list(spatial_graph).to(device)

        model.train()
        min_loss = np.inf
        patience = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_params = model.state_dict()

        for epoch in range(epochs):
            train_loss = 0.0
            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            z, neg_z, summary = model(expr, edge_list)
            loss = model.loss(z, neg_z, summary)

            z1, z2 = torch.index_select(z, 0, edge_list[0, :]), torch.index_select(z, 0, edge_list[1, :])
            coords = torch.tensor(adata_filtered.obsm['spatial']).float().to(device)
            c1, c2 = torch.index_select(coords, 0, edge_list[0, :]), torch.index_select(coords, 0, edge_list[1, :])

            pdist = torch.nn.PairwiseDistance(p=2)

            z_dists = pdist(z1, z2)
            z_dists = z_dists / torch.max(z_dists)

            sp_dists = pdist(c1, c2)
            sp_dists = sp_dists / torch.max(sp_dists)

            penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), z_dists.size(dim=0)).to(device)
            loss = loss + spatial_regularization_strength * penalty_1

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if train_loss > min_loss:
                patience += 1
            else:
                patience = 0
                min_loss = train_loss
                best_params = model.state_dict()
            if epoch % 10 == 1:
                print("Epoch %d/%d" % (epoch + 1, epochs))
                print("Loss:" + str(train_loss))
            if patience > max_patience and epoch > min_stop:
                break

        model.load_state_dict(best_params)

        z, _, _ = model(expr, edge_list)
        embedding = z.cpu().detach().numpy()
        save_dir = os.path.dirname(embedding_save_filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savetxt(embedding_save_filepath, embedding[:, :], delimiter="\t")
        print(f"Training is completed!\nEmbedding is saved at {embedding_save_filepath}")
        return embedding

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphEncoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.prelu2 = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        return x