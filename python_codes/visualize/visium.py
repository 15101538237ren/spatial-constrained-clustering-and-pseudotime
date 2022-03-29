# -*- coding: utf-8 -*-
import os, math, shutil, json
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr
import matplotlib.patches as patches
#from python_codes.train.train import train
from python_codes.train.clustering import clustering
from python_codes.train.pseudotime import pseudotime
from python_codes.util.util import load_stereo_seq_data, preprocessing_data, save_preprocessed_data, load_preprocessed_data, save_features
import warnings
from python_codes.train.clustering import clustering
from python_codes.train.pseudotime import pseudotime
from python_codes.util.exchangeable_loom import write_exchangeable_loom
warnings.filterwarnings("ignore")
from python_codes.util.util import *
from matplotlib.colors import to_hex
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial','Roboto']
rcParams['savefig.dpi'] = 300
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
title_sz = 16

VISIUM_DATASETS = ["V1_Human_Lymph_Node",
                   "V1_Breast_Cancer_Block_A_Section_1",
                   "V1_Mouse_Brain_Sagittal_Anterior",
                   "V1_Mouse_Brain_Sagittal_Posterior",
                   "V1_Adult_Mouse_Brain_Coronal_Section_1",
                   "Parent_Visium_Human_Cerebellum"]


####################################
#----------Get Annotations---------#
####################################

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

def figure(nrow, ncol, rsz=3., csz=3., wspace=.4, hspace=.5, left=None, right=None, bottom=None):
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * csz, nrow * rsz))
    plt_setting()
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom)
    return fig, axs

def plot_annotation(args, adata, sample_name, nrow = 1, ncol=3, rsz=2.5, csz=2.8, wspace=.4, hspace=.5, left=None, right=None, alpha=1.):
    alpha = .8 if sample_name == "V1_Human_Lymph_Node" else alpha
    fig, axs = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right)
    expr_dir = os.path.join(args.dataset_dir, "Visium", sample_name)
    scale_factor_fp = os.path.join(expr_dir, "spatial", "scalefactors_json.json")
    with open(scale_factor_fp, "r") as json_file:
        data_dict = json.load(json_file)
        scale = data_dict["tissue_lowres_scalef"]

    spatial_cords = adata.obsm['spatial'].astype(float) * scale
    x, y = spatial_cords[:, 0], spatial_cords[:, 1]
    img = plt.imread(os.path.join(expr_dir, "spatial", "tissue_lowres_image.png"))
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xscale, yscale = xmax - xmin, ymax - ymin
    max_scale = max(xscale, yscale)
    scale_offset = max_scale * .55
    center_x, center_y = (xmin + xmax)/2., (ymin + ymax)/2.
    xlim, ylim = [center_x - scale_offset, center_x + scale_offset], [center_y - scale_offset, center_y + scale_offset]
    for ax in axs:
        ax.axis('off')
        ax.imshow(img, alpha=alpha)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.invert_yaxis()
    return fig, axs, x, y, xlim, ylim

def plot_clustering(args, adata, sample_name, method="leiden", dataset="Visium", cm = plt.get_cmap("Paired"), scatter_sz=1., nrow= 1):
    original_spatial = args.spatial
    fig, axs, x, y, xlim, ylim = plot_annotation(args, adata, sample_name, nrow=nrow, ncol=3, rsz=4, csz=4, wspace=.4, hspace=.5, left=.1, right=.95)
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
        # ax.invert_yaxis()
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

def plot_pseudotime(args, adata, sample_name, dataset="Visium", cm = plt.get_cmap("gist_rainbow"), scatter_sz=1.3, nrow = 1):
    original_spatial = args.spatial
    fig, axs, x, y, xlim, ylim = plot_annotation(args, adata, sample_name, nrow=nrow, ncol=3, rsz=4, csz=4, wspace=.4, hspace=.5, left=.1, right=.95)
    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        ax = axs[sid + 1]
        args.spatial = spatial
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
        pseudotimes = pd.read_csv(f"{output_dir}/pseudotime.tsv", header=None).values.flatten().astype(float)
        st = ax.scatter(x, y, s=scatter_sz, c=pseudotimes, cmap=cm, marker=".")
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

def plot_clustering_and_pseudotime(args, adata, sample_name, method="leiden", dataset="Visium", cluster_cm = plt.get_cmap("Paired"),pseudotime_cm = plt.get_cmap("gist_rainbow"), scatter_sz=1., nrow= 1, title_padding=10):
    original_spatial = args.spatial
    fig, axs, x, y, xlim, ylim = plot_annotation(args, adata, sample_name, nrow=nrow, ncol=3, rsz=2.5, csz=3, wspace=0.3, hspace=.05, left=.1, right=.95)

    args.spatial = True

    ax = axs[1] # clustering
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(int)
    uniq_pred = np.unique(pred_clusters)
    n_cluster = len(uniq_pred)
    for cid, cluster in enumerate(uniq_pred):
        color = cluster_cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
        ind = pred_clusters == cluster
        ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster, marker=".")
    ax.set_facecolor("none")

    ax = axs[-1]  # pseudotime
    pseudotimes = pd.read_csv(f"{output_dir}/pseudotime.tsv", header=None).values.flatten().astype(float)
    st = ax.scatter(x, y, s=scatter_sz, c=pseudotimes, cmap=pseudotime_cm, marker=".")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = fig.colorbar(st, cax=cax)
    clb.ax.set_ylabel("pSM value", labelpad=10, rotation=270, fontsize=10, weight='bold')
    # title = "pSM"
    # ax.set_title(title, fontsize=title_sz, pad=title_padding)
    ax.set_facecolor("none")

    fig_fp = f"{output_dir}/segmentation_and_pSM.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')
    args.spatial = original_spatial

def train_pipeline(args, adata_filtered, spatial_graph, sample_name, dataset="Visium", clustering_method="leiden", resolution=.6, n_neighbors=10, isTrain=True):
    for spatial in [False, True]:
        args.spatial = spatial
        if isTrain:
            embedding = train(args, adata_filtered, spatial_graph)
            save_features(args, embedding, dataset, sample_name)

        clustering(args, dataset, sample_name, clustering_method, n_neighbors=n_neighbors, resolution=resolution)
        pseudotime(args, dataset, sample_name, n_neighbors=n_neighbors, resolution=resolution)

def basic_pipeline(args):
    for dataset in VISIUM_DATASETS:
        print(f'===== Data: {dataset} =====')
        adata = load_visium_data(args, dataset)
        adata_filtered, spatial_graph = preprocessing_data(args, adata)
        #train_pipeline(args, adata_filtered, spatial_graph, dataset, n_neighbors=6, isTrain=True)
        # plot_clustering(args, adata_filtered, dataset, scatter_sz=1)
        # plot_pseudotime(args, adata_filtered, dataset, scatter_sz=1)
        plot_clustering_and_pseudotime(args, adata_filtered, dataset, scatter_sz=1)