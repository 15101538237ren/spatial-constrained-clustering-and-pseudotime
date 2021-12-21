# -*- coding:utf-8 -*-
import math
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
# from python_codes.train.train import train
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
    cluster_color_dict = get_cluster_colors(args, sample_name)
    unique_cluster_dict = {cluster:cluster_color_dict[cluster]["abbr"] for cluster in cluster_color_dict.keys()}
    uniq_pred = np.unique(pred_clusters)
    for cid, cluster in enumerate(uniq_pred):
        pred_clusters[pred_clusters == cluster] = unique_cluster_dict[int(cluster)]
    return pred_clusters

def get_cluster_colors_and_labels_original():
    ann_dict = {
        0: "Cancer 1",
        1: "Immune:B/plasma",
        2: "Adipose",
        3: "Immune:APC/B/T cells",
        4: "Cancer:Immune rich",
        5: "Cancer 2",
        6: "Cancer Connective"
    }
    color_dict = {
        0: "#771122",
        1: "#AA4488",
        2: "#05C1BA",
        3: "#F7E54A",
        4: "#D55802",
        5: "#137777",
        6: "#124477"
    }
    return ann_dict, color_dict

def get_cluster_colors(args, sample_name):
    fp = f'{args.dataset_dir}/Visium/Breast_Cancer/putative_cell_type_colors/{sample_name}.csv'
    df = pd.read_csv(fp)
    clusters = df["Cluster ID"].values.astype(int)
    annotations = df["Annotations"].values.astype(str)
    colors = df["Color"].values.astype(str)
    abbrs = df["Abbr"].values.astype(str)
    cur_dict = {}
    for cid, cluster in enumerate(clusters):
        cur_dict[cluster] = {
            "annotation" : annotations[cid],
            "color" : colors[cid],
            "abbr" : abbrs[cid]
        }
    return cur_dict

def get_top_n_cluster_specific_genes(args,  sample_name, method, dataset="breast_cancer", top_n=3):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    cluster_marker_genes_fp = f'{output_dir}/marker_genes_pval_gby_{method}.tsv'
    df = pd.read_csv(cluster_marker_genes_fp, sep="\t")
    df = df.loc[:top_n, df.columns.str.endswith("_n")]
    genes = df.values.flatten().astype(str)
    return genes

def save_cluster_specific_genes(args, adata, sample_name, method, dataset="breast_cancer", qval=0.05):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    fp = f'{args.dataset_dir}/Visium/Breast_Cancer/putative_cell_type_colors/{sample_name}.csv'
    df = pd.read_csv(fp)
    abbrs = np.array(np.unique(df["Abbr"].values.astype(str)))

    cluster_marker_genes_fp = f'{output_dir}/marker_genes_pval_gby_{method}.tsv'
    df = pd.read_csv(cluster_marker_genes_fp, sep="\t", header=0)


    for cid, cluster_name in enumerate(abbrs):
        sub_df = df.loc[df.loc[:, f"{cluster_name}_p"] <= qval, f"{cluster_name}_n"]
        genes = np.array(np.unique(sub_df.values.flatten().astype(str)))
        output_fp = f'{output_dir}/cluster_specific_marker_genes/{cluster_name}.tsv'
        mkdir(os.path.dirname(output_fp))
        np.savetxt(output_fp, genes[:], delimiter="\n", fmt="%s")
        print(f"Saved at {output_fp}")

    all_genes = np.array(list(adata.var_names))
    output_fp = f'{output_dir}/cluster_specific_marker_genes/background_genes.tsv'
    mkdir(os.path.dirname(output_fp))
    np.savetxt(output_fp, all_genes[:], delimiter="\n", fmt="%s")
    print(f"Saved at {output_fp}")
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

def plot_hne_and_annotation(args, adata, sample_name, nrow = 1, scale = 0.045, ncol=4, rsz=2.5, csz=2.8, wspace=.4, hspace=.5, annotation=True):
    fig, axs = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace)
    if nrow == 1:
        for ax in axs:
            ax.axis('off')
    ax = axs[0]
    if annotation:
        fp = f'{args.dataset_dir}/Visium/Breast_Cancer/ST-pat/img/{sample_name[0]}1_annotated.png'
    else:
        fp = f'{args.dataset_dir}/Visium/Breast_Cancer/ST-imgs/{sample_name[0]}/{sample_name}/HE.jpg'
    img = plt.imread(fp)
    ax.imshow(img)
    ax.set_title("H & E", fontsize=title_sz)
    x, y = adata.obsm["spatial"][:, 0]*scale, adata.obsm["spatial"][:, 1]*scale
    if not annotation:
        xlim = [np.min(x), np.max(x) * 1.05]
        ylim = [np.min(y) * .75, np.max(y) * 1.1]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        xlim, ylim = None, None
    ax.invert_yaxis()
    return fig, axs, x, y, img, xlim, ylim

def plot_clustering(args, adata, sample_name, method="leiden", dataset="breast_cancer", cm = plt.get_cmap("Paired"), scale = .62, scatter_sz=1.3, nrow = 1, annotation=True):
    original_spatial = args.spatial
    fig, axs, x, y, img, xlim, ylim = plot_hne_and_annotation(args, adata, sample_name, scale=scale, nrow=nrow, ncol=3, rsz=2.6, csz=3.2, wspace=.3, hspace=.4, annotation=annotation)
    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        ax = axs[sid + 1]
        args.spatial = spatial
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
        pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(int)
        uniq_pred = np.unique(pred_clusters)
        n_cluster = len(uniq_pred)
        if not annotation:
            ax.imshow(img)
        for cid, cluster in enumerate(uniq_pred):
            color = cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
            ind = pred_clusters == cluster
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster)
        title = args.arch if not spatial else "%s + SP" % args.arch
        ax.set_title(title, fontsize=title_sz, pad=-30)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
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
    args.spatial = original_spatial

def plot_pseudotime(args, adata, sample_name, dataset="breast_cancer", cm = plt.get_cmap("gist_rainbow"), scale = 0.62, scatter_sz=1.3, nrow = 1):
    original_spatial = args.spatial
    fig, axs, x, y, img, _, _ = plot_hne_and_annotation(args, adata, sample_name, scale=scale, nrow=nrow, ncol=3)
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

def plot_clustering_and_pseudotime(args, adata, sample_name, method="leiden", dataset="breast_cancer", scale = 1., scatter_sz=1.3, nrow = 1, annotation=False):
    original_spatial = args.spatial
    args.spatial = True
    fig, axs, x, y, img, xlim, ylim = plot_hne_and_annotation(args, adata, sample_name, scale=scale, nrow=nrow, ncol=4, rsz=2.6,
                                                              csz=3.9, wspace=1, hspace=.4, annotation=annotation)
    ax = axs[1]
    ax.imshow(img)
    # fp = f'{args.dataset_dir}/Visium/Breast_Cancer/ST-pat/img/{sample_name[0]}1_annotated.png'
    # img2 = plt.imread(fp)
    # ax.imshow(img2)
    fp = f'{args.dataset_dir}/Visium/Breast_Cancer/ST-cluster/lbl/{sample_name}-cluster-annotation.tsv'
    df = pd.read_csv(fp, sep="\t")
    coords = df[["pixel_x", "pixel_y"]].values.astype(float)
    pred_clusters = df["label"].values.astype(int)
    cluster_dict, color_dict = get_cluster_colors_and_labels_original()
    uniq_pred = np.unique(pred_clusters)
    uniq_pred = sorted(uniq_pred, key=lambda cluster: cluster_dict[cluster])
    for cid, cluster in enumerate(uniq_pred):
        ind = pred_clusters == cluster
        ax.scatter(coords[ind, 0], coords[ind, 1], s=scatter_sz, color=color_dict[cluster], label=cluster_dict[cluster])
    ax.set_title("Annotation", fontsize=title_sz)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.0, 0.5))

    ax = axs[2]
    ax.imshow(img)
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(str)
    uniq_pred = np.unique(pred_clusters)
    cluster_color_dict = get_cluster_colors(args, sample_name)
    unique_cluster_dict = {cluster: cluster_color_dict[cluster]["abbr"] for cluster in cluster_color_dict.keys()}
    color_dict_for_cluster = {}
    for cid, cluster in enumerate(uniq_pred):
        label = unique_cluster_dict[int(cluster)]
        color_dict_for_cluster[label] = f"#{cluster_color_dict[int(cluster)]['color']}"
        pred_clusters[pred_clusters == cluster] = label
    uniq_pred = sorted(np.unique(pred_clusters))
    for cid, cluster in enumerate(uniq_pred):
        ind = pred_clusters == cluster
        ax.scatter(x[ind], y[ind], s=scatter_sz, color=color_dict_for_cluster[cluster], label=cluster)

    ax.set_title("Clustering", fontsize=title_sz)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()

    ax = axs[3]
    ax.imshow(img)

    pseudotimes = pd.read_csv(f"{output_dir}/pseudotime.tsv", header=None).values.flatten().astype(float)
    pseudo_time_cm = plt.get_cmap("gist_rainbow")
    st = ax.scatter(x, y, s=scatter_sz, c=pseudotimes, cmap=pseudo_time_cm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%")
    clb = fig.colorbar(st, cax=cax)
    clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=8, weight='bold')
    ax.set_title("Pseudotime", fontsize=title_sz)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()

    fig_fp = f"{output_dir}/cluster+pseudotime.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')
    args.spatial = original_spatial

def plot_rank_marker_genes_group(args, sample_name, adata_filtered, method="cluster", dataset="breast_cancer", top_n_genes=3):
    original_spatial = args.spatial
    args.spatial = True
    pred_clusters = get_clusters(args, sample_name)
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    adata_filtered.obs[method] = pd.Categorical(pred_clusters)
    sc.tl.rank_genes_groups(adata_filtered, method, method='wilcoxon')
    sc.pl.rank_genes_groups(adata_filtered, n_genes=25, ncols=5, fontsize=10, sharey=False, save=f"{sample_name}_ranks_gby_{method}.pdf")
    sc.pl.rank_genes_groups_heatmap(adata_filtered, n_genes=top_n_genes, standard_scale='var',  show_gene_labels=True, save=f"{sample_name}_heatmap_gby_{method}.pdf")
    sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=top_n_genes, standard_scale='var', cmap='bwr', save=f"{sample_name}_mean_expr_gby_{method}.pdf")
    sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=top_n_genes, values_to_plot="logfoldchanges", cmap='bwr', vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change', save=f"{sample_name}_dot_lfc_gby_{method}.pdf")
    sc.pl.rank_genes_groups_matrixplot(adata_filtered, n_genes=top_n_genes, values_to_plot="logfoldchanges", cmap='bwr', vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change', save=f"{sample_name}_matrix_lfc_gby_{method}.pdf")
    sc.pl.rank_genes_groups_matrixplot(adata_filtered, n_genes=top_n_genes, cmap='bwr', colorbar_title='Mean Expr.', save=f"{sample_name}_matrix_mean_expr_gby_{method}.pdf")

    files = [f"rank_genes_groups_cluster{sample_name}_ranks_gby_{method}.pdf",
             f"heatmap{sample_name}_heatmap_gby_{method}.pdf",
             f"dotplot_{sample_name}_mean_expr_gby_{method}.pdf",
             f"dotplot_{sample_name}_dot_lfc_gby_{method}.pdf",
             f"matrixplot_{sample_name}_matrix_lfc_gby_{method}.pdf",
             f"matrixplot_{sample_name}_matrix_mean_expr_gby_{method}.pdf"]
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

def plot_expr_in_ST(args, adata, genes, sample_name, dataset="breast_cancer", scatter_sz= 6., cm = plt.get_cmap("magma"), n_cols = 5):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    mkdir(output_dir)
    n_genes = len(genes)
    n_rows = int(math.ceil(n_genes/n_cols))
    fig, axs = figure(n_rows, n_cols, rsz=2.2, csz=3., wspace=.2, hspace=.2)
    exprs = adata.X
    all_genes = np.array(list(adata.var_names))

    fp = f'{args.dataset_dir}/Visium/Breast_Cancer/ST-imgs/{sample_name[0]}/{sample_name}/HE.jpg'
    img = plt.imread(fp)
    x, y = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    xlim = [np.min(x), np.max(x) * 1.05]
    ylim = [np.min(y) * .75, np.max(y) * 1.1]
    for gid, gene in enumerate(genes):
        row = gid // n_cols
        col = gid % n_cols
        ax = axs[row][col] if n_rows > 1 else axs[col]
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        expr = exprs[:, all_genes == gene]
        # expr /= np.max(expr)
        ax.imshow(img)
        st = ax.scatter(x, y, s=scatter_sz, c=expr, cmap=cm)#
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * .9, box.height])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.invert_yaxis()
        # if col == n_cols - 1:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = fig.colorbar(st, cax=cax)
        clb.ax.set_ylabel("Expr.", labelpad=10, rotation=270, fontsize=10, weight='bold')
        ax.set_title(gene, fontsize=12)
    fig_fp = f"{output_dir}/ST_expr.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')


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

def basic_pipeline(args):
    letters = ["G"]#["A", "B", "C", "D", "E", "F", "G", "H"]#["H"]#
    n_samples = [1 for letter in letters] #[6, 6, 6, 6, 3, 3, 3, 3]#
    sample_list = [f"{letter}{sid}" for lid, letter in enumerate(letters) for sid in range(1, n_samples[lid] + 1)]

    for sample_name in sample_list:
        adata = load_breast_cancer_data(args, sample_name)
        adata_filtered, genes, cells = train_pipeline(args, adata, sample_name, n_neighbors=5, isTrain=False)
        # plot_clustering(args, adata, sample_name, scatter_sz=3, annotation=False, scale=1)
        # plot_pseudotime(args, adata, sample_name)
        plot_rank_marker_genes_group(args, sample_name, adata_filtered, top_n_genes=5)

def figure_pipeline(args):
    sample_list = ["G1"]
    for sample_name in sample_list:
        adata = load_breast_cancer_data(args, sample_name)
        plot_clustering_and_pseudotime(args, adata, sample_name, scatter_sz=5)
        # adata_filtered, genes, cells = train_pipeline(args, adata, sample_name, n_neighbors=5)
        # plot_clustering(args, adata, sample_name, scatter_sz=3, annotation=False, scale=1)
        # plot_pseudotime(args, adata, sample_name)


def expr_analysis_pipeline(args):
    sample_list = ["G1"]
    for sample_name in sample_list:
        adata = load_breast_cancer_data(args, sample_name)
        save_cluster_specific_genes(args, adata, sample_name, "cluster")
        # top_n_cluster_specific_genes = get_top_n_cluster_specific_genes(args, sample_name, "cluster", top_n=2)
        # # top_n_cluster_specific_genes = top_n_cluster_specific_genes[:5]
        # plot_expr_in_ST(args, adata, top_n_cluster_specific_genes, sample_name, scatter_sz=2)