# -*- coding:utf-8 -*-
import phate
import anndata
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

def plot_hne_and_annotation(args, adata, sample_name, nrow = 1, HnE=False, cm = plt.get_cmap("plasma_r"), scale = 0.045, scatter_sz=1.3, ncol=4, rsz=2.5, csz=2.8, wspace=.4, hspace=.5):
    subfig_offset = 1 if HnE else 0
    fig, axs = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace)
    if nrow == 1:
        for ax in axs:
            ax.axis('off')
    ax = axs[subfig_offset]

    fp = f'{args.dataset_dir}/Visium/Breast_Cancer/ST-pat/img/{sample_name[0]}1_annotated.png'
    img = plt.imread(fp)
    ax.imshow(img)
    ax.set_title("Annotation", fontsize=title_sz, pad=-30)
    x, y = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    return fig, axs, x, y, subfig_offset

def plot_clustering(args, adata, sample_name, method="leiden", dataset="breast_cancer", HnE=False, cm = plt.get_cmap("tab20"), scale = 1., scatter_sz=1.3, nrow = 1):
    original_spatial = args.spatial
    fig, axs, x, y, subfig_offset = plot_hne_and_annotation(args, adata, sample_name, HnE=HnE, cm=cm, scale=scale, scatter_sz=scatter_sz, nrow=nrow, ncol=3)
    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        ax = axs[subfig_offset + sid + 1]
        args.spatial = spatial
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
        pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(int)
        uniq_pred = np.unique(pred_clusters)
        n_cluster = len(uniq_pred)
        for cid, cluster in enumerate(uniq_pred):
            color = cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
            ind = pred_clusters == cluster
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster)
        title = args.arch if not spatial else "%s + SP" % args.arch
        ax.set_title(title, fontsize=title_sz, pad=-30)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.1,
                         borderaxespad=.1, ncol=2)
        for handle in lgnd.legendHandles:
            handle._sizes = [8]
    fig_fp = f"{output_dir}/{method}.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')
    args.spatial = original_spatial

def plot_pseudotime(args, sample_name, dataset="breast_cancer", HnE=False, cm = plt.get_cmap("gist_rainbow"), scale = 0.045):
    original_spatial = args.spatial
    fig, axs, x, y, subfig_offset = plot_hne_and_annotation(args, sample_name, HnE=HnE, cm=cm, scale=scale, nrow=1, ncol=3)
    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        ax = axs[subfig_offset + sid + 1]
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

def train_pipeline(args, adata, sample_name, dataset="breast_cancer", clustering_method="leiden", resolution = .8, n_neighbors = 10, isTrain=True):
    adata_filtered, expr, genes, cells, spatial_graph, spatial_dists = preprocessing_data(args, adata)
    if isTrain:
        embedding = train(args, expr, spatial_graph, spatial_dists)
        save_features(args, embedding, dataset, sample_name)
        clustering(args, dataset, sample_name, clustering_method, n_neighbors=n_neighbors, resolution=resolution)
        # pseudotime(args, dataset, sample_name, root_cell_type="Epi-epithelial cells", cell_types=cell_types, n_neighbors=n_neighbors, resolution=resolution)
    return adata_filtered, genes, cells

def basic_pipeline(args):
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    n_samples = [1 for letter in letters] #[6, 6, 6, 6, 3, 3, 3, 3]
    sample_list = [f"{letter}{sid}" for lid, letter in enumerate(letters) for sid in range(1, n_samples[lid] + 1)]

    for sample_name in sample_list:
        adata = load_breast_cancer_data(args, sample_name)
        # train_pipeline(args, adata, sample_name)
        plot_clustering(args, adata, sample_name)
