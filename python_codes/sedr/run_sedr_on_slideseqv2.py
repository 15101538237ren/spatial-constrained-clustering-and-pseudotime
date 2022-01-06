import os
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from python_codes.util.config import args
from python_codes.util.util import load_slideseqv2_data, preprocessing_data, save_preprocessed_data, load_preprocessed_data
from python_codes.sedr.graph_func import graph_construction
from python_codes.sedr.utils_func import mk_dir, adata_preprocess
from python_codes.sedr.SEDR_train import SEDR_Train

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)

# ################ Parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--cell_feat_dim', type=int, default=200, help='Dim of PCA')
parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.')

params = parser.parse_args()
params.device = device

# ################ Path setting
dataset = "slideseq_v2"
clustering_method = "leiden"
sample_list = ['slideseq_v2']
args.dataset_dir = f'../../data'
args.output_dir = f'../../output'
max_cells = 8000
for sample_idx, sample_name in enumerate(sample_list):
    print(f'===== Data {sample_idx + 1} : {sample_name}')
    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}/preprocessed'
    indices_fp = os.path.join(data_root, "indices-for-sedr.npy")
    if os.path.exists(indices_fp):
        with open(indices_fp, 'rb') as f:
            indices = np.load(f)
            print("loaded indices successful!")
        adata_filtered, spatial_graph = load_preprocessed_data(args, dataset, sample_name, sedr=True)
    else:
        adata = load_slideseqv2_data()
        indices = np.random.choice(adata.shape[0], max_cells, replace=False)
        with open(indices_fp, 'wb') as f:
            np.save(f, indices)
        print("Saved indices")
        adata = adata[indices, :]
        adata_filtered, spatial_graph = preprocessing_data(args, adata)
        save_preprocessed_data(args, dataset, sample_name, adata, spatial_graph, sedr=True)

    sc.pp.pca(adata_filtered, n_comps=params.cell_feat_dim)
    graph_dict = graph_construction(spatial_graph, adata_filtered.shape[0], params)

    params.save_path = f'{args.output_dir}/{dataset}/{sample_name}/sedr'
    mk_dir(params.save_path)
    params.cell_num = adata_filtered.shape[0]
    print('==== Graph Construction Finished')

    # ################## Model training
    expr = adata_filtered.X.todense() if type(adata_filtered.X).__module__ != np.__name__ else adata_filtered.X
    sedr_net = SEDR_Train(expr, graph_dict, params)
    if params.using_dec:
        sedr_net.train_with_dec()
    else:
        sedr_net.train_without_dec()
    embeddings, _, _, _ = sedr_net.process()

    np.savez(f'{params.save_path}/sedr_embedding.npz', sedr_feat=embeddings, params=params)

    # ################## Result plot
    adata_sedr = anndata.AnnData(embeddings)
    adata_sedr.uns['spatial'] = adata_filtered.uns['spatial']
    adata_sedr.obsm['spatial'] = adata_filtered.obsm['spatial']

    sc.pp.neighbors(adata_sedr, n_neighbors=params.eval_graph_n)
    sc.tl.leiden(adata_sedr, key_added="sedr_leiden", resolution=1.0)

    df_meta = pd.DataFrame(adata_sedr.obs['sedr_leiden'], columns=["cluster"])
    df_meta.to_csv(f'{params.save_path}/metadata.tsv', sep='\t', index=False)
