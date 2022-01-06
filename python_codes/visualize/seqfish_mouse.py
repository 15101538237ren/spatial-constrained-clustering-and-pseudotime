# -*- coding: utf-8 -*-
import os
import warnings
# from python_codes.train.train import train
from python_codes.train.clustering import clustering
from python_codes.train.pseudotime import pseudotime
from python_codes.util.util import load_seqfish_mouse_data, preprocessing_data, save_preprocessed_data, load_preprocessed_data, save_features
from python_codes.train.clustering import clustering
from python_codes.train.pseudotime import pseudotime
warnings.filterwarnings("ignore")
from scipy.sparse import csr_matrix
from python_codes.util.util import *
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial','Roboto']
rcParams['savefig.dpi'] = 300
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
from python_codes.util.exchangeable_loom import write_exchangeable_loom
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

def plot_annotation(args, adata, nrow = 1, scale = 0.045, ncol=4, rsz=2.5, csz=2.8, wspace=.4, hspace=.5, scatter_sz=1.):
    fig, axs = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace)
    if nrow == 1:
        for ax in axs:
            ax.axis('off')
    ax = axs[0]
    x, y = adata.obsm["spatial"][:, 0]*scale, adata.obsm["spatial"][:, 1]*scale
    prefix = "celltype_mapped_refined"
    annotated_cell_types = adata.obs[prefix]
    cell_type_strs = annotated_cell_types.cat.categories.astype(str)
    cell_type_ints = annotated_cell_types.values.codes
    cell_type_colors = list(adata.uns[f'{prefix}_colors'].astype(str))
    # colors = np.array([cell_type_colors[item] for item in cell_type_ints])
    cm = plt.get_cmap("tab20")
    n_cluster = len(cell_type_colors)
    for cid in range(n_cluster):
        cit = cell_type_ints == cid
        color = cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
        ax.scatter(x[cit], y[cit], s=scatter_sz, color=color, label=cell_type_strs[cid], marker=".")
    ax.set_facecolor("none")
    ax.set_title("Annotation", fontsize=title_sz)
    xlim, ylim = None, None
    ax.invert_yaxis()
    return fig, axs, x, y, xlim, ylim

def plot_clustering(args, adata, sample_name, method="leiden", dataset="seqfish_mouse", cm = plt.get_cmap("tab20"), scale = .62, scatter_sz=1., nrow= 1):
    original_spatial = args.spatial
    fig, axs, x, y, xlim, ylim = plot_annotation(args, adata, scale=scale, nrow=nrow, ncol=3, rsz=5, csz=5.5, wspace=.3, hspace=.4)
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
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster, marker=".")
        ax.set_facecolor("none")
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

def plot_pseudotime(args, adata, sample_name, dataset="seqfish_mouse", cm = plt.get_cmap("gist_rainbow"), scale = 0.62, scatter_sz=1.3, nrow = 1):
    original_spatial = args.spatial
    fig, axs, x, y, _, _ = plot_annotation(args, adata, scale=scale, nrow=nrow, ncol=3, rsz=5, csz=5.5, wspace=.3, hspace=.4)
    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        ax = axs[sid + 1]
        args.spatial = spatial
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
        pseudotimes = pd.read_csv(f"{output_dir}/pseudotime.tsv", header=None).values.flatten().astype(float)
        st = ax.scatter(x, y, s=scatter_sz, c=pseudotimes, cmap=cm, marker=".")
        ax.invert_yaxis()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = fig.colorbar(st, cax=cax)
        clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
        title = args.arch if not spatial else "%s + SP" % args.arch
        ax.set_title(title, fontsize=title_sz)
        ax.set_facecolor("none")
    fig_fp = f"{output_dir}/psudotime.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')
    args.spatial = original_spatial


####################################
#-------------Pipelines------------#
####################################

def export_data_pipeline(args):
    dataset = "seqfish_mouse"
    data_root = f'{args.dataset_dir}/{dataset}/{dataset}/export'
    mkdir(data_root)

    adata = load_seqfish_mouse_data()
    write_exchangeable_loom(adata,f'{data_root}/adata.loom')

    locs = pd.DataFrame(adata.obsm["spatial"], columns=["x", "y"])
    locs.to_csv(f"{data_root}/locs.tsv", sep="\t", index=False)
    print(f'===== Exported {dataset} =====')

def train_pipeline(args, adata_filtered, spatial_graph, sample_name, dataset="seqfish_mouse", clustering_method="leiden", resolution=.3, n_neighbors=15, isTrain=True):
    for spatial in [False, True]:
        args.spatial = spatial
        if isTrain:
            embedding = train(args, adata_filtered, spatial_graph)
            save_features(args, embedding, dataset, sample_name)
        # clustering(args, dataset, sample_name, clustering_method, n_neighbors=n_neighbors, resolution=resolution)
        pseudotime(args, dataset, sample_name, root_cell_type=None, cell_types=None, n_neighbors=n_neighbors,
                   resolution=resolution)

def basic_pipeline(args):
    dataset = "seqfish_mouse"

    print(f'===== Data: {dataset} =====')
    data_root = f'{args.dataset_dir}/{dataset}/{dataset}/preprocessed'
    if os.path.exists(f"{data_root}/adata.h5ad"):
        adata_filtered, spatial_graph = load_preprocessed_data(args, dataset, dataset)
    else:
        adata = load_seqfish_mouse_data()
        adata_filtered, spatial_graph = preprocessing_data(args, adata)
        save_preprocessed_data(args, dataset, dataset, adata, spatial_graph)

    train_pipeline(args, adata_filtered, spatial_graph, dataset, n_neighbors=30, isTrain=False)
    # plot_clustering(args, adata_filtered, dataset, scatter_sz=1.5, scale=1)
    plot_pseudotime(args, adata_filtered, dataset, scatter_sz=1.5, scale=1)