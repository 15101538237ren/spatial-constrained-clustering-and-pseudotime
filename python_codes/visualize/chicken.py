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

def plot_hne_and_annotation(args, sample_name, nrow = 1, HnE=False, cm = plt.get_cmap("plasma_r"), scale = 0.045, scatter_sz=1.3, ncol=4, rsz=2.5, csz=2.8, wspace=.4, hspace=.5):
    subfig_offset = 1 if HnE else 0

    data_root = f'{args.dataset_dir}/Visium/Chicken_Dev/ST/{sample_name}'
    adata = load_ST_file(data_root)
    coord = adata.obsm['spatial'].astype(float) * scale
    x, y = coord[:, 0], coord[:, 1]
    anno_clusters = get_annotations_chicken(args, sample_name)
    img = plt.imread(f"{data_root}/spatial/tissue_lowres_image.png")
    xlimits = [0, 550]
    ylimits = [0, 530]
    fig, axs = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace)
    if nrow == 1:
        for ax in axs:
            ax.axis('off')
            # ax.set_xlim(xlimits)
            # ax.set_ylim(ylimits)
            ax.invert_yaxis()
        ax = axs[subfig_offset]
    else:
        for axr in axs:
            for ax in axr:
                ax.axis('off')
                ax.set_xlim(xlimits)
                ax.set_ylim(ylimits)
                ax.invert_yaxis()
        ax = axs[0][subfig_offset]
    ax.set_title("Annotation", fontsize= title_sz, pad=-30)
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
    return fig, axs, x, y, subfig_offset

def plot_clustering(args, sample_name, method="leiden", dataset="chicken", HnE=False, cm = plt.get_cmap("tab20"), scale = 0.045, scatter_sz=1.3, nrow = 1):
    original_spatial = args.spatial
    fig, axs, x, y, subfig_offset = plot_hne_and_annotation(args, sample_name, HnE=HnE, cm=cm, scale=scale, scatter_sz=scatter_sz, nrow=nrow, ncol=3)
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
    fig_fp = f"{output_dir}/{method}.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')
    args.spatial = original_spatial

def plot_pseudotime(args, sample_name, dataset="chicken", HnE=False, cm = plt.get_cmap("gist_rainbow"), scale = 0.045):
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
    #
    # for sample_name in sample_list:
    #     anno_clusters = get_annotations_chicken(args, sample_name)
    #     for spatial in [False, True]:
    #         args.spatial = spatial
    #         adata = load_chicken_data(args, sample_name)
    #         adata_filtered, expr, genes, cells, spatial_graph, spatial_dists = preprocessing_data(args, adata)
    #         embedding = train(args, expr, spatial_graph, spatial_dists)
    #         save_features(args, embedding, dataset, sample_name)
    #         clustering(args, dataset, sample_name, clustering_method, n_neighbors=n_neighbors, resolution=resolution)
    #         pseudotime(args, dataset, sample_name, root_cell_type="Epi-epithelial cells", cell_types=anno_clusters,
    #                    n_neighbors=n_neighbors, resolution=resolution)
    #     plot_clustering(args, sample_name, clustering_method)
    #     plot_pseudotime(args, sample_name)