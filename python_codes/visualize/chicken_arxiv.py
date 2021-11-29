# -*- coding:utf-8 -*-
import phate
import anndata
import warnings
warnings.filterwarnings("ignore")
from python_codes.util.util import *
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial','Roboto']
rcParams['savefig.dpi'] = 300
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
title_sz = 16

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


def get_annotations_chicken(args, sample_name):
    data_root = f'{args.dataset_dir}/Visium/Chicken_Dev/ST/{sample_name}'
    df_meta = pd.read_csv(f"{data_root}/metadata.csv")
    anno_clusters = df_meta['celltype_prediction'].values.astype(str)
    return anno_clusters

def get_offsets_x(max_xs):
    cur_sum = 0
    offsets = []
    for offset in max_xs:
        offsets.append(cur_sum)
        cur_sum += offset
    offsets = [int(item * 1.2) for item in offsets]
    return offsets

def offset_adata(adatas, offsets):
    for aid in range(len(adatas)):
        adatas[aid].obsm['spatial'][:, 0] += offsets[aid]
    return adatas

def get_merged_chicken_data(args, sample_list):
    adatas = [load_chicken_data(args, sample_name) for sample_name in sample_list]
    annotations = [get_annotations_chicken(args, sample_name) for sample_name in sample_list]
    n_samples = [anno.shape[0] for anno in annotations]
    anno_days = np.concatenate([np.array([sample_list[sid] for _ in range(n_sample)]) for sid, n_sample in enumerate(n_samples)], axis=0)
    annotations = np.concatenate(annotations, axis=0)
    max_xs = [np.max(adata.obsm['spatial'][:, 0]) for adata in adatas]
    offset_x = get_offsets_x(max_xs)
    adatas = offset_adata(adatas, offset_x)
    adata = anndata.concat(adatas, axis=0)
    return adata, annotations, anno_days

def plot_hne_and_annotation_merged(args, adata, anno_clusters, anno_days, n_comparison= 2, cm = plt.get_cmap("gist_rainbow"), scale= 0.045, scatter_sz=1.2, rsz=3.5, csz=14, wspace=.4, hspace=.4):
    nrow = 1 + n_comparison
    ncol = 1
    fig, axs = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace)
    for ax in axs:
        ax.axis('off')
    ax = axs[0]

    coord = adata.obsm['spatial'].astype(float) * scale
    x, y = coord[:, 0], coord[:, 1]

    ax.set_title("Annotation", fontsize= title_sz, pad=-10)
    uniq_annots = [cluster for cluster in np.unique(anno_clusters) if cluster != 'nan']
    n_cluster = len(uniq_annots)
    for cid, cluster in enumerate(uniq_annots):
        color = cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
        ind = anno_clusters == cluster
        ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.1, borderaxespad=.1)
    for handle in lgnd.legendHandles:
        handle._sizes = [8]
    return fig, axs, x, y

def plot_phate(args, expr, anno_clusters, dataset="chicken", sample_name="merged", cm = plt.get_cmap("tab20"), scatter_sz=1.2):
    original_spatial = args.spatial
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    args.spatial = original_spatial
    phate_op = phate.PHATE(k=15, t=150)
    data_phate = phate_op.fit_transform(expr)
    fig_fp = f"{output_dir}/phate.pdf"
    phate.plot.scatter2d(data_phate, c=anno_clusters, cmap=cm, s=4, filename=fig_fp, legend_title="Cell Types", dpi=300, figsize=(8,8),
                      ticks=False, label_prefix="PHATE")

def plot_clustering_merged(args, adata, anno_clusters, anno_days, method="leiden", dataset="chicken", cm = plt.get_cmap("gist_rainbow"), scale = 0.045, scatter_sz=1.3):
    original_spatial = args.spatial
    fig, axs, x, y = plot_hne_and_annotation_merged(args, adata, anno_clusters, anno_days, cm=cm, scale=scale, scatter_sz=scatter_sz)
    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        ax = axs[sid + 1]
        args.spatial = spatial
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, "merged")}'
        pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(int)
        uniq_pred = np.unique(pred_clusters)
        n_cluster = len(uniq_pred)
        for cid, cluster in enumerate(uniq_pred):
            color = cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
            ind = pred_clusters == cluster
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster)
        title = args.arch if not spatial else "%s + SP" % args.arch
        ax.set_title(title, fontsize=title_sz)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.1,
        #                  borderaxespad=.1, ncol=2)
        # for handle in lgnd.legendHandles:
        #     handle._sizes = [8]
    fig_fp = f"{output_dir}/{method}.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')
    args.spatial = original_spatial

def plot_pseudotime_merged(args, adata, anno_clusters, anno_days, dataset="chicken", cm = plt.get_cmap("gist_rainbow"), scale = 0.045):
    original_spatial = args.spatial
    fig, axs, x, y = plot_hne_and_annotation_merged(args, adata, anno_clusters, anno_days, cm=cm, scale=scale)
    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        ax = axs[sid + 1]
        args.spatial = spatial
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, "merged")}'
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


def pipeline():
    pass
    # -*- coding: utf-8 -*-
    # from python_codes.util.config import args
    # from python_codes.train.train import train
    # from python_codes.train.clustering import clustering
    # from python_codes.train.pseudotime import pseudotime
    # from python_codes.visualize.chicken import *
    # from python_codes.util.util import load_chicken_data, preprocessing_data, save_features
    # dataset = "chicken"
    # clustering_method = "leiden"
    # resolution = 1.0
    # n_neighbors = 6
    # sample_list = ['D4', 'D7', 'D10', 'D14']
    # adata, anno_clusters, anno_days = get_merged_chicken_data(args, sample_list)
    # adata_filtered, expr, genes, cells, spatial_graph, spatial_dists = preprocessing_data(args, adata)
    #
    # plot_phate(args, adata.X, anno_clusters)
    #
    # # sample_name = "merged"
    # # for spatial in [False, True]:
    # #     args.spatial = spatial
    # #     # embedding = train(args, expr, spatial_graph, spatial_dists)
    # #     # save_features(args, embedding, dataset, sample_name)
    # #     clustering(args, dataset, sample_name, clustering_method, n_neighbors=n_neighbors, resolution=resolution)
    # #     #pseudotime(args, dataset, sample_name, root_cell_type="Epi-epithelial cells", cell_types=anno_clusters, n_neighbors=n_neighbors, resolution=resolution)
    # # plot_clustering_merged(args, adata_filtered, anno_clusters, anno_days, clustering_method)
    # # plot_pseudotime_merged(args, adata_filtered, anno_clusters, anno_days)