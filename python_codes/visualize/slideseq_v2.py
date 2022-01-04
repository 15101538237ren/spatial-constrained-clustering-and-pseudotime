# -*- coding: utf-8 -*-
import os
# from python_codes.train.train import train
from python_codes.train.clustering import clustering
from python_codes.train.pseudotime import pseudotime
from python_codes.util.util import load_slideseqv2_data, preprocessing_data, save_preprocessed_data, load_preprocessed_data, save_features
import warnings
from python_codes.train.clustering import clustering
from python_codes.train.pseudotime import pseudotime
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

def plot_annotation(args, adata, sample_name, nrow = 1, scale = 0.045, ncol=4, rsz=2.5, csz=2.8, wspace=.4, hspace=.5, scatter_sz=1., annotation=True):
    fig, axs = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace)
    if nrow == 1:
        for ax in axs:
            ax.axis('off')
    ax = axs[0]
    x, y = adata.obsm["spatial"][:, 0]*scale, adata.obsm["spatial"][:, 1]*scale
    annotated_cell_types = adata.obs["cluster"]
    cell_type_strs = annotated_cell_types.cat.categories.astype(str)
    cell_type_ints = annotated_cell_types.values.codes
    cell_type_colors = list(adata.uns['cluster_colors'].astype(str))
    colors = np.array([cell_type_colors[item] for item in cell_type_ints])
    for cid in range(len(cell_type_colors)):
        cit = cell_type_ints == cid
        ax.scatter(x[cit], y[cit], s=scatter_sz, c=colors[cit], label=cell_type_strs[cid], marker=".")
    ax.set_facecolor("none")
    ax.set_title("Annotation", fontsize=title_sz)
    xlim, ylim = None, None
    ax.invert_yaxis()
    return fig, axs, x, y, xlim, ylim


def plot_clustering(args, adata, sample_name, method="leiden", dataset="slideseq_v2", cm = plt.get_cmap("Paired"), scale = .62, scatter_sz=1., nrow = 1, annotation=True):
    original_spatial = args.spatial
    fig, axs, x, y, xlim, ylim = plot_annotation(args, adata, sample_name, scale=scale, nrow=nrow, ncol=3, rsz=3, csz=3.5, wspace=.3, hspace=.4, annotation=annotation)
    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        ax = axs[sid + 1]
        args.spatial = spatial
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
        pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(int)
        uniq_pred = np.unique(pred_clusters)
        n_cluster = len(uniq_pred)
        for cid, cluster in enumerate(uniq_pred):
            color = cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
            ind = pred_clusters == cluster
            sc = ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster, marker=".")
            sc.set_facecolor("none")
        title = args.arch if not spatial else "%s + SP" % args.arch
        ax.set_title(title, fontsize=title_sz, pad=-30)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.invert_yaxis()
        # box = ax.get_position()
        # height_ratio = 1.0
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * height_ratio])
        # lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.1,
        #                  borderaxespad=.1)
        # for handle in lgnd.legendHandles:
        #     handle._sizes = [8]
    fig_fp = f"{output_dir}/{method}.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')
    args.spatial = original_spatial

def plot_pseudotime(args, adata, sample_name, dataset="slideseq_v2", cm = plt.get_cmap("gist_rainbow"), scale = 0.62, scatter_sz=1.3, nrow = 1):
    original_spatial = args.spatial
    fig, axs, x, y, _, _ = plot_annotation(args, adata, sample_name, scale=scale, nrow=nrow, ncol=3)
    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        ax = axs[sid + 1]
        args.spatial = spatial
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
        pseudotimes = pd.read_csv(f"{output_dir}/pseudotime.tsv", header=None).values.flatten().astype(float)
        st = ax.scatter(x, y, s=scatter_sz, c=pseudotimes, cmap=cm, marker=".")
        st.set_facecolor("none")
        ax.invert_yaxis()
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

def train_pipeline(args, adata_filtered, spatial_graph, sample_name, dataset="slideseq_v2", clustering_method="leiden", resolution=.8, n_neighbors=20, isTrain=True):
    for spatial in [False, True]:
        args.spatial = spatial
        if isTrain:
            embedding = train(args, adata_filtered, spatial_graph)
            save_features(args, embedding, dataset, sample_name)
        clustering(args, dataset, sample_name, clustering_method, n_neighbors=n_neighbors, resolution=resolution)
        # pseudotime(args, dataset, sample_name, root_cell_type=None, cell_types=None, n_neighbors=n_neighbors,
        #            resolution=resolution)

def basic_pipeline(args):
    dataset = "slideseq_v2"
    clustering_method = "leiden"
    sample_list = ['slideseq_v2']

    for sample_idx, sample_name in enumerate(sample_list):
        print(f'===== Data {sample_idx + 1} : {sample_name}')
        data_root = f'{args.dataset_dir}/{dataset}/{sample_name}/preprocessed'
        if os.path.exists(data_root):
            adata_filtered, spatial_graph = load_preprocessed_data(args, dataset, sample_name)
        else:
            adata = load_slideseqv2_data()
            adata_filtered, spatial_graph = preprocessing_data(args, adata)
            save_preprocessed_data(args, dataset, sample_name, adata, spatial_graph)

        train_pipeline(args, adata_filtered, spatial_graph, sample_name, n_neighbors=20, isTrain=False)
        plot_clustering(args, adata_filtered, sample_name, scatter_sz=1, annotation=False, scale=1, method=clustering_method)
        # plot_pseudotime(args, adata_filtered, sample_name, scatter_sz=1, scale=1)
        # plot_rank_marker_genes_group(args, sample_name, adata_filtered, top_n_genes=5)
        # get_correlation_matrix_btw_clusters(args, sample_name, adata_filtered)