# -*- coding: utf-8 -*-
import os
import scanpy as sc
import numpy as np
from scipy.spatial import distance_matrix
from python_codes.util.config import args
from python_codes.util.util import load_datasets, preprocessing_data, save_preprocessed_data, load_preprocessed_data, get_target_fp
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

def plot_annotation(args, adata, sample_name, nrow = 1, scale = 0.045, ncol=4, rsz=2.5, csz=2.8, wspace=.4, hspace=.5, scatter_sz=1.):
    fig, ax = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace)
    ax.axis('off')
    x, y = adata.obsm["spatial"][:, 0]*scale, adata.obsm["spatial"][:, 1]*scale
    xlim, ylim = None, None
    return fig, ax, x, y, xlim, ylim


def res_search_fixed_clus(clustering_method, adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=False):
        if clustering_method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs[clustering_method]).leiden.unique())
        else:
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(np.unique(pd.DataFrame(adata.obs[clustering_method].cat.codes.values).values.flatten()))
        print("Try resolution %3f found %d clusters: target %d" % (res, count_unique, fixed_clus_count))
        if count_unique == fixed_clus_count:
            print("Found resolution:" + str(res))
            return res
        elif count_unique > fixed_clus_count:
            print("Found resolution: %.3f" % (res - increment))
            return res - increment

def scanpy_clustering(args, adata, dataset, sample_name, method = "leiden", n_neighbors=50, ncluster = 8):
    output_dir = f'{args.output_dir}/{dataset}/{sample_name}/scanpy'
    mkdir(output_dir)
    cluster_fp = os.path.join(output_dir, f"{method}.tsv")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    resolution = res_search_fixed_clus(method, adata, ncluster)
    sc.tl.leiden(adata, resolution=float(resolution))
    labels = adata.obs["leiden"].cat.codes
    np.savetxt(cluster_fp, labels, fmt='%d', header='', footer='', comments='')
    print("Saved %s succesful!" % cluster_fp)

def scanpy_pseudotime(args, adata, dataset, sample_name, n_neighbors=20, root_cell_type = None, cell_types=None, resolution=1.0):
    output_dir = f'{args.output_dir}/{dataset}/{sample_name}/scanpy'
    mkdir(output_dir)
    pseudotime_fp = os.path.join(output_dir, "pseudotime.tsv")

    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution)
    sc.tl.paga(adata)
    indices = np.arange(adata.shape[0])
    selected_ind = np.random.choice(indices, 5000, False)
    #sub_adata_x = np.array(adata[selected_ind, :].X.todense()).astype(float)
    sub_adata_x = np.array(adata[selected_ind, :].X).astype(float)
    sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
    adata.uns['iroot'] = np.argmax(sum_dists)
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)
    pseudotimes = adata.obs['dpt_pseudotime'].to_numpy()
    np.savetxt(pseudotime_fp, pseudotimes, fmt='%.5f', header='', footer='', comments='')
    print("Saved %s succesful!" % pseudotime_fp)

def scanpy_pca(args, adata, dataset, sample_name, n_comps=2):
    output_dir = f'{args.output_dir}/{dataset}/{sample_name}/scanpy'
    mkdir(output_dir)
    pca_fp = os.path.join(output_dir, "PCA.tsv")

    sc.pp.pca(adata, n_comps=n_comps)
    PCs = adata.obsm['X_pca']
    np.savetxt(pca_fp, PCs, fmt='%.5f\t%.5f', header='', footer='', comments='')
    print("Saved %s succesful!" % pca_fp)


def plot_clustering(args, adata, sample_name, dataset, method="leiden", cm= plt.get_cmap("tab20"), scale=.62, scatter_sz=1., nrow= 1):
    fig, ax, x, y, xlim, ylim = plot_annotation(args, adata, sample_name, scale=scale, nrow=nrow, ncol=1, rsz=5, csz=6, wspace=.3, hspace=.4)
    output_dir = f'{args.output_dir}/{dataset}/{sample_name}/scanpy'
    pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(int)
    uniq_pred = np.unique(pred_clusters)
    n_cluster = len(uniq_pred)
    for cid, cluster in enumerate(uniq_pred):
        color = cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
        ind = pred_clusters == cluster
        if dataset == "stereo_seq":
            ax.scatter(-y[ind], x[ind], s=scatter_sz, color=color, label=cluster, marker=".")
        else:
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster, marker=".")
    ax.set_facecolor("none")
    ax.invert_yaxis()
    box = ax.get_position()
    height_ratio = 1.0
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * height_ratio])
    lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.1,
                     borderaxespad=.1)
    for handle in lgnd.legendHandles:
        handle._sizes = [8]
    fig_fp = f"{output_dir}/{method}.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_pseudotime(args, adata, sample_name, dataset, cm = plt.get_cmap("gist_rainbow"), scale = 0.62, scatter_sz=1.3, nrow = 1):
    fig, ax, x, y, xlim, ylim = plot_annotation(args, adata, sample_name, scale=scale, nrow=nrow, ncol=1, rsz=5,
                                                csz=5.5, wspace=.3, hspace=.4)
    output_dir = f'{args.output_dir}/{dataset}/{sample_name}/scanpy'
    pseudotimes = pd.read_csv(f"{output_dir}/pseudotime.tsv", header=None).values.flatten().astype(float)
    st = ax.scatter(x, y, s=scatter_sz, c=pseudotimes, cmap=cm, marker=".")
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = fig.colorbar(st, cax=cax)
    clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    title = "Scanpy"
    ax.set_title(title, fontsize=title_sz)
    ax.set_facecolor("none")
    fig_fp = f"{output_dir}/psudotime.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def basic_pipeline(args):
    args.dataset_dir = f'../../data'
    args.output_dir = f'../../output'
    n_neighbors = 15
    datasets = ["stereo_seq", "slideseq_v2","seqfish_mouse"]#] #,
    for dataset in datasets:
        print(f'===== Data {dataset} =====')
        data_root = f'{args.dataset_dir}/{dataset}/{dataset}/preprocessed'
        if os.path.exists(f"{data_root}/adata.h5ad"):
            adata_filtered, spatial_graph = load_preprocessed_data(args, dataset, dataset)
        else:
            adata = load_datasets(args, dataset)
            adata_filtered, spatial_graph = preprocessing_data(args, adata)
        print(adata_filtered.shape)
        #     save_preprocessed_data(args, dataset, dataset, adata_filtered, spatial_graph)
        # sc.tl.pca(adata_filtered, svd_solver='arpack')
        # scanpy_clustering(args, adata_filtered, dataset, dataset, "leiden", n_neighbors=n_neighbors, ncluster=8)
        #scanpy_pseudotime(args, adata_filtered, dataset, dataset, n_neighbors=n_neighbors, resolution=resolution)
        #scanpy_pca(args, adata_filtered, dataset, dataset)
        # plot_clustering(args, adata_filtered, dataset, dataset, scatter_sz=1.5, scale=1)
        # plot_pseudotime(args, adata_filtered, dataset, dataset, scatter_sz=1.5, scale=1)

if __name__ == "__main__":
    basic_pipeline(args)