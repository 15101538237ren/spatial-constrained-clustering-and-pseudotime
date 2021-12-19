# -*- coding:utf-8 -*-
import phate
import anndata
import shutil
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from python_codes.train.train import train
from python_codes.train.clustering import clustering
from python_codes.train.pseudotime import pseudotime
from python_codes.util.util import load_breast_cancer_data, preprocessing_data, save_features
warnings.filterwarnings("ignore")
from python_codes.util.util import *
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial','Roboto']
rcParams['savefig.dpi'] = 300
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
title_sz = 16

####################################
#----------Get Annotations---------#
####################################

def get_clusters(args, sample_name, method="leiden", dataset="breast_cancer"):
    original_spatial = args.spatial
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(str)
    args.spatial = original_spatial
    return pred_clusters

####################################
#-------------Plotting-------------#
####################################
def plt_setting():
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 30
    plt.rc('font', size=MEDIUM_SIZE, weight="bold")  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def figure(nrow, ncol, rsz=3., csz=3., wspace=.4, hspace=.5):
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * csz, nrow * rsz))
    plt_setting()
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    return fig, axs

def plot_hne_and_annotation(args, adata, sample_name, nrow = 1, cm = plt.get_cmap("plasma_r"), scale = 0.045, scatter_sz=1.3, ncol=4, rsz=2.5, csz=2.8, wspace=.4, hspace=.5):
    fig, axs = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace)
    if nrow == 1:
        for ax in axs:
            ax.axis('off')
    ax = axs[0]

    fp = f'{args.dataset_dir}/Visium/Breast_Cancer/ST-pat/img/{sample_name[0]}1_annotated.png'
    img = plt.imread(fp)
    ax.imshow(img)
    ax.set_title("Annotation", fontsize=title_sz, pad=-30)
    x, y = adata.obsm["spatial"][:, 0]*scale, adata.obsm["spatial"][:, 1]*scale
    return fig, axs, x, y, img

def plot_clustering(args, adata, sample_name, method="leiden", dataset="breast_cancer", cm = plt.get_cmap("tab20"), scale = .62, scatter_sz=1.3, nrow = 1):
    original_spatial = args.spatial
    fig, axs, x, y, img = plot_hne_and_annotation(args, adata, sample_name, cm=cm, scale=scale, scatter_sz=scatter_sz, nrow=nrow, ncol=3)
    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        ax = axs[sid + 1]
        args.spatial = spatial
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
        pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(int)
        uniq_pred = np.unique(pred_clusters)
        n_cluster = len(uniq_pred)
        # ax.imshow(img)
        ax.invert_yaxis()
        for cid, cluster in enumerate(uniq_pred):
            color = cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
            ind = pred_clusters == cluster
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster)
        title = args.arch if not spatial else "%s + SP" % args.arch
        ax.set_title(title, fontsize=title_sz, pad=-30)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
        lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.1,
                         borderaxespad=.1)
        for handle in lgnd.legendHandles:
            handle._sizes = [8]
    fig_fp = f"{output_dir}/{method}.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')
    args.spatial = original_spatial

def plot_pseudotime(args, adata, sample_name, dataset="breast_cancer", cm = plt.get_cmap("gist_rainbow"), scale = 0.62, scatter_sz=1.3, nrow = 1):
    original_spatial = args.spatial
    fig, axs, x, y, img = plot_hne_and_annotation(args, adata, sample_name, cm=cm, scale=scale, scatter_sz=scatter_sz, nrow=nrow, ncol=3)
    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        ax = axs[sid + 1]
        ax.imshow(img)
        args.spatial = spatial
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
        pseudotimes = pd.read_csv(f"{output_dir}/pseudotime.tsv", header=None).values.flatten().astype(float)
        st = ax.scatter(x, y, s=1, c=pseudotimes, cmap=cm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = fig.colorbar(st, cax=cax)
        clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
        title = args.arch if not spatial else "%s + SP" % args.arch
        ax.set_title(title, fontsize=title_sz)
    fig_fp = f"{output_dir}/psudotime.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')
    args.spatial = original_spatial


####################################
#-------------Pipelines------------#
####################################

def train_pipeline(args, adata, sample_name, dataset="breast_cancer", clustering_method="leiden", resolution=.5, n_neighbors=6, isTrain=True):
    adata_filtered, expr, genes, cells, spatial_graph, spatial_dists = preprocessing_data(args, adata)
    if isTrain:
        for spatial in [False, True]:
            args.spatial = spatial
            embedding = train(args, expr, spatial_graph, spatial_dists)
            save_features(args, embedding, dataset, sample_name)
            clustering(args, dataset, sample_name, clustering_method, n_neighbors=n_neighbors, resolution=resolution)
            pseudotime(args, dataset, sample_name, root_cell_type=None, cell_types=None, n_neighbors=n_neighbors, resolution=resolution)
    return adata_filtered, genes, cells

def plot_rank_marker_genes_group(args, sample_name, adata_filtered, method="cluster", dataset="breast_cancer", top_n_genes=3):
    original_spatial = args.spatial
    args.spatial = True
    pred_clusters = get_clusters(args, sample_name)
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    adata_filtered.obs[method] = pd.Categorical(pred_clusters)
    sc.tl.rank_genes_groups(adata_filtered, method, method='wilcoxon')
    sc.pl.rank_genes_groups(adata_filtered, n_genes=25, ncols=5, fontsize=10, sharey=False, save=f"{sample_name}_ranks_gby_{method}.pdf")
    sc.pl.rank_genes_groups_heatmap(adata_filtered, n_genes=top_n_genes, standard_scale='var',  show_gene_labels=True, save=f"{sample_name}_heatmap_gby_{method}.pdf")
    sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=top_n_genes, standard_scale='var', save=f"{sample_name}_mean_expr_gby_{method}.pdf")
    sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=top_n_genes, values_to_plot="logfoldchanges", cmap='bwr', vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change', save=f"{sample_name}_dot_lfc_gby_{method}.pdf")
    sc.pl.rank_genes_groups_matrixplot(adata_filtered, n_genes=top_n_genes, values_to_plot="logfoldchanges", cmap='bwr', vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change', save=f"{sample_name}_matrix_lfc_gby_{method}.pdf")

    files = [f"rank_genes_groups_cluster{sample_name}_ranks_gby_{method}.pdf",
             f"heatmap{sample_name}_heatmap_gby_{method}.pdf",
             f"dotplot_{sample_name}_mean_expr_gby_{method}.pdf",
             f"dotplot_{sample_name}_dot_lfc_gby_{method}.pdf",
             f"matrixplot_{sample_name}_matrix_lfc_gby_{method}.pdf"]
    for file in files:
        src_fp = f"./figures/{file}"
        target_fp = f"{output_dir}/{file}"
        shutil.move(src_fp, target_fp)
    args.spatial = original_spatial
    cluster_marker_genes_fp = f'{output_dir}/marker_genes_pval_gby_{method}.tsv'
    mkdir(os.path.dirname(cluster_marker_genes_fp))
    result = adata_filtered.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    df = pd.DataFrame(
        {group + '_' + key[:1]: result[key][group]
        for group in groups for key in ['names', 'pvals']})
    df.to_csv(cluster_marker_genes_fp, sep="\t", index=False)


def basic_pipeline(args):
    letters = ["A", "B", "C", "D", "E", "F", "G"]#, "H"]#["H"]#
    n_samples = [1 for letter in letters] #[6, 6, 6, 6, 3, 3, 3, 3]#
    sample_list = [f"{letter}{sid}" for lid, letter in enumerate(letters) for sid in range(1, n_samples[lid] + 1)]

    for sample_name in sample_list:
        adata = load_breast_cancer_data(args, sample_name)
        adata_filtered, genes, cells = train_pipeline(args, adata, sample_name, n_neighbors=5)
        plot_clustering(args, adata, sample_name, scatter_sz=2)
        plot_pseudotime(args, adata, sample_name)
        plot_rank_marker_genes_group(args, sample_name, adata_filtered)
