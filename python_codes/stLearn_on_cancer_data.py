import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
                            homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scanpy as sc
import stlearn as st
from pathlib import Path, PurePath
from typing import Optional, Union
from anndata import AnnData
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from python_codes.util.config import args
from python_codes.util.util import load_breast_cancer_data, mkdir
from matplotlib.image import imread
from scipy.spatial import distance_matrix

dataset = "Visium"
library_id="Breast_Cancer"
data_root = f'{args.dataset_dir}/{dataset}/{library_id}'

# def Read10X(
#     img_file: str,
#     library_id: str = None,
#     quality="hires",
#     scale=1.0
# ) -> AnnData:
#     sample_name = "G1"
#     base_dir = f"{args.dataset_dir}/Visium/Breast_Cancer"
#     count_fp = f'{base_dir}/ST-cnts/{sample_name}.tsv'
#     adata = sc.read_csv(count_fp, delimiter='\t', first_column_names=True)
#     coord_fp = f'{base_dir}/ST-spotfiles/{sample_name}_selection.tsv'
#     coord_df = pd.read_csv(coord_fp, delimiter='\t')
#     spots_idx_dicts = {f"{item[0]}x{item[1]}": idx for idx, item in enumerate(coord_df[["x", "y"]].values.astype(int))}
#     spots_selected = np.array(
#         [sid for sid, spot in enumerate(list(adata.obs_names)) if spot in spots_idx_dicts]).astype(int)
#     coords = coord_df[["pixel_x", "pixel_y"]].values
#     adata.obsm["spatial"] = np.array([coords[spots_idx_dicts[spot]] for spot in adata.obs_names])
#     adata = adata[spots_selected, :]
#     adata.var_names_make_unique()
#     coord_df = coord_df.iloc[spots_selected, :]
#     adata.obs["array_row"] = coord_df["x"].values.astype(int)
#     adata.obs["array_col"] = coord_df["y"].values.astype(int)
#     adata.uns["spatial"] = dict()
#     adata.uns["spatial"][library_id] = dict()
#     adata.uns["spatial"][library_id]['images'] = dict()
#     adata.uns["spatial"][library_id]['images'][quality] = imread(img_file)
#     image_coor = adata.obsm["spatial"] * scale
#     adata.obs["imagecol"] = image_coor[:, 0]
#     adata.obs["imagerow"] = image_coor[:, 1]
#     adata.uns["spatial"][library_id]["use_quality"] = quality
#     return adata
#
# adata = Read10X(img_file=f"{data_root}/ST-imgs/G/G1/HE.jpg",library_id=library_id)
# st.pp.filter_genes(adata,min_cells=1)
# st.pp.normalize_total(adata)
# st.pp.log1p(adata)
# # pre-processing for spot image
# TILE_PATH = Path(f"{data_root}/tiles")
# TILE_PATH.mkdir(parents=True, exist_ok=True)
# st.pp.tiling(adata, TILE_PATH)
# st.pp.extract_feature(adata)
#
# st.em.run_pca(adata, n_comps=50)
# data_SME = adata.copy()
# # apply stSME to normalise log transformed data
# st.spatial.SME.SME_normalize(data_SME, use_data="raw")
# data_SME.X = data_SME.obsm['raw_SME_normalized']
# st.pp.scale(data_SME)
# st.em.run_pca(data_SME, n_comps=50)
#
# df_PCA = pd.DataFrame(data= data_SME.obsm['X_pca'], index= data_SME.obs.index)
out_dir = f"{args.output_dir}/{library_id}/G1/stLearn"
mkdir(out_dir)
# df_PCA.to_csv(f"{out_dir}/PCs.tsv", sep='\t', header=False)

adata = sc.read_csv(f"{out_dir}/PCs.tsv", delimiter="\t", first_column_names=True)
sc.pp.neighbors(adata, n_neighbors=6, use_rep='X')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=.5)
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
np.savetxt(f"{out_dir}/pseudotime.tsv", pseudotimes, fmt='%.5f', header='', footer='', comments='')
print("Saved %s succesful!" % f"{out_dir}/pseudotime.tsv")


