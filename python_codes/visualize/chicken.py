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
from python_codes.util.util import load_chicken_data, preprocessing_data, save_features
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
#-----------Figure Setting----------#
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

####################################
#----------Get Annotations---------#
####################################

def get_annotations_chicken(args, sample_name):
    data_root = f'{args.dataset_dir}/Visium/Chicken_Dev/ST/{sample_name}'
    df_meta = pd.read_csv(f"{data_root}/metadata.csv")
    anno_clusters = df_meta['celltype_prediction'].values.astype(str)
    region_annos = df_meta['region'].values.astype(str)
    return anno_clusters, region_annos

def get_cluster_annotations_chicken(args, samples, dataset="chicken", method="leiden"):
    args.spatial = True
    annotation_colors = get_cluster_colors(args, samples)
    cluster_annotations = []
    for sid, sample in enumerate(samples):
        cluster_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample)}'
        pred_clusters = pd.read_csv(f"{cluster_dir}/{method}.tsv", header=None).values.flatten().astype(int)
        annotations = np.array([annotation_colors[sid][cluster][0] for cluster in pred_clusters]).astype(str)
        cluster_annotations.append(annotations)
    return cluster_annotations

def get_cell_type_annotation_colors():
    annotations_names = ["Fibroblast cells", "Epi-epithelial cells",
                   "Immature myocardial cells", "Endocardial cells",
                   "Cardiomyocytes-1", "Cardiomyocytes-2",
                   "MT-enriched cardiomyocytes",
                   "Vascular endothelial cells",
                   "Valve cells", "Mural cells",
                   "Erythrocytes", "Macrophages",
                   "TMSB4X high cells"]
    annotation_colors = {
        "Fibroblast cells": "#DBD301",
        "Epi-epithelial cells": "#008856",
        "Immature myocardial cells": "#F6A500",
        "Endocardial cells": "#604D96",
        "Cardiomyocytes-1": "#B4476E",
        "Cardiomyocytes-2": "#F99379",
        "MT-enriched cardiomyocytes": "#F38400",
        "Vascular endothelial cells": "#C2B280",
        "Valve cells": "#0466A5",
        "Mural cells": "#BD012E",
        "Erythrocytes": "#848482",
        "Macrophages": "#875692",
        "TMSB4X high cells": "#E691AD"
    }
    return annotations_names, annotation_colors

def get_region_annotation_colors():
    annotations_names = ["Valves", "Atria", "Outflow tract",
                         "Epicardium", "Epicardium- like",
                         "Endothelium", "Ventricle", "Right ventricle",
                         "Trabecular LV and endocardium",
                         "Compact LV and inter-ventricular septum"]
    annotation_colors = {
        "Valves": "#BB509D",
        "Atria": "#C7A1CA",
        "Outflow tract": "#FAAD15",
        "Epicardium": "#73C8B3",
        "Epicardium- like": "#B01E6B",
        "Endothelium": "#90AD3D",
        "Ventricle": "#ED2630",
        "Right ventricle": "#4D79BA",
        "Trabecular LV and \nendocardium": "#67BA44",
        "Compact LV and \ninter-ventricular septum": "#48B0D1"
    }
    return annotations_names, annotation_colors

def get_cluster_colors(args, time_points):
    data_root = f'{args.dataset_dir}/Visium/Chicken_Dev/putative_cell_type_colors'
    annotation_colors = []
    for day in time_points:
        df = pd.read_csv(f"{data_root}/{day}.csv")
        clusters = df["Cluster ID"].values.astype(int)
        annotations = df["Annotations"].values.astype(str)
        colors = df["Color"].values.astype(str)
        cur_dict = {}
        for cid, cluster in enumerate(clusters):
            cur_dict[cluster] = [annotations[cid], colors[cid]]
        annotation_colors.append(cur_dict)

    return annotation_colors

def get_cluster_colors_dict(args):
    days = ['D4', 'D7', 'D10', 'D14']
    data_root = f'{args.dataset_dir}/Visium/Chicken_Dev/putative_cell_type_colors'
    annotation_colors = {}
    for day in days:
        df = pd.read_csv(f"{data_root}/{day}.csv")
        clusters = df["Cluster ID"].values.astype(int)
        annotations = df["Annotations"].values.astype(str)
        colors = df["Color"].values.astype(str)
        for cid, cluster in enumerate(clusters):
            annotation_colors[annotations[cid]] = f"#{colors[cid]}"
    return annotation_colors

def get_day_colors(cm = plt.get_cmap("Spectral")):
    days = ['D4', 'D7', 'D10', 'D14']
    n_day = len(days)
    color_dict = {day: cm((did * (n_day / (n_day - 1.0))) / n_day) for did, day in enumerate(days)}
    return color_dict


def get_lineage_clusters_and_color_dict(args, lineage, dataset="chicken", spatial=True, cluster_method="leiden", cm = plt.get_cmap("tab20")):
    args.spatial = spatial
    feature_dir = f'{args.output_dir}/{get_target_fp(args, dataset, lineage)}'
    cluster_fp = os.path.join(feature_dir, f"{cluster_method}.tsv")
    pred_clusters = pd.read_csv(cluster_fp, header=None).values.flatten().astype(str)
    uniq_pred = np.unique(pred_clusters)
    n_cluster = uniq_pred.size
    color_dict = {cluster: cm((did * (n_cluster / (n_cluster - 1.0))) / n_cluster) for did, cluster in enumerate(uniq_pred)}
    return pred_clusters, color_dict

def get_phate(args, lineage, dataset="chicken"):
    args.spatial = True
    feature_dir = f'{args.output_dir}/{get_target_fp(args, dataset, lineage)}'
    feature_fp = os.path.join(feature_dir, "features.tsv")
    adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)

    phate_op = phate.PHATE(k=5, t=100, gamma=0)
    data_phate = phate_op.fit_transform(adata.X)
    return phate_op, data_phate

def get_phate_by_adata(adata):
    phate_op = phate.PHATE(k=5, t=100, gamma=0)
    data_phate = phate_op.fit_transform(adata.X)
    return phate_op, data_phate

####################################
#----------Data Processing---------#
####################################

def filter_adatas_annotations_chicken(adatas, cluster_annotations, cell_types, regions, days, lineages):

    filtered_adatas, filtered_annotations, filtered_cell_types, filtered_regions, filtered_days = [], [], [], [], []

    for lineage in lineages:
        for cid, annotations in enumerate(cluster_annotations):
            ind = np.array([True if annotation.startswith(lineage) else False for annotation in annotations])
            filtered_adatas.append(adatas[cid][ind, :])
            filtered_annotations.append(annotations[ind])
            filtered_cell_types.append(cell_types[cid][ind])
            filtered_regions.append(regions[cid][ind])
            filtered_days.append(days[cid][ind])

    return filtered_adatas, filtered_annotations, filtered_cell_types, filtered_regions, filtered_days

def merge_adatas_annotations_chicken(adatas, cluster_annotations, cell_types, regions, days):
    merged_adata = anndata.concat(adatas, axis=0)
    merged_cluster_annotations = np.concatenate(cluster_annotations, axis=0)
    merged_cell_types = np.concatenate(cell_types, axis=0)
    merged_regions = np.concatenate(regions, axis=0)
    merged_days = np.concatenate(days, axis=0)
    return merged_adata, merged_cluster_annotations, merged_cell_types, merged_regions, merged_days

####################################
#-------------Plotting-------------#
####################################

def plot_hne_and_annotation(args, sample_name, nrow = 1, HnE=False, cm = plt.get_cmap("plasma_r"), scale = 0.045, scatter_sz=1.3, ncol=4, rsz=2.5, csz=2.8, wspace=.4, hspace=.5):
    subfig_offset = 1 if HnE else 0

    data_root = f'{args.dataset_dir}/Visium/Chicken_Dev/ST/{sample_name}'
    adata = load_ST_file(data_root)
    coord = adata.obsm['spatial'].astype(float) * scale
    x, y = coord[:, 0], coord[:, 1]
    anno_clusters, region_annos = get_annotations_chicken(args, sample_name)
    fig, axs = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace)
    if nrow == 1:
        for ax in axs:
            ax.axis('off')
    ax = axs[subfig_offset]
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

def plot_phate(args, sample_name, expr, anno_clusters, dataset="chicken", cm = plt.get_cmap("gist_rainbow"), scale = 0.045):
    original_spatial = args.spatial
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    args.spatial = original_spatial
    phate_op = phate.PHATE(k=10, t=150, gamma=0)
    data_phate = phate_op.fit_transform(expr)
    fig_fp = f"{output_dir}/phate.pdf"
    phate.plot.scatter2d(data_phate, c=anno_clusters, cmap=cm, s=6, filename=fig_fp, legend_title="Cell Types", dpi=300,
                         figsize=(8, 8),
                         ticks=False, label_prefix="PHATE")

def get_clusters(args, sample_name, method="leiden", dataset="chicken"):
    original_spatial = args.spatial
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(str)
    args.spatial = original_spatial
    return pred_clusters

def plot_rank_marker_genes_group(args, sample_name, adata_filtered, group_values, method="cluster", dataset="chicken"):
    original_spatial = args.spatial
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    adata_filtered.obs[method] = pd.Categorical(group_values)
    sc.tl.rank_genes_groups(adata_filtered, method, method='wilcoxon')
    sc.pl.rank_genes_groups(adata_filtered, n_genes=25, ncols=5, fontsize=10, sharey=False, save=f"{sample_name}_ranks_gby_{method}.pdf")
    sc.pl.rank_genes_groups_heatmap(adata_filtered, n_genes=3, standard_scale='var', save=f"{sample_name}_heatmap_gby_{method}.pdf")
    sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=3, standard_scale='var', save=f"{sample_name}_mean_expr_gby_{method}.pdf")
    sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=3, values_to_plot="logfoldchanges", cmap='bwr', vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change', save=f"{sample_name}_dot_lfc_gby_{method}.pdf")
    sc.pl.rank_genes_groups_matrixplot(adata_filtered, n_genes=3, values_to_plot="logfoldchanges", cmap='bwr', vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change', save=f"{sample_name}_matrix_lfc_gby_{method}.pdf")
    args.spatial = original_spatial
    cluster_marker_genes_fp = f'{output_dir}/marker_genes_pval_gby_{method}.tsv'
    result = adata_filtered.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    df = pd.DataFrame(
        {group + '_' + key[:1]: result[key][group]
        for group in groups for key in ['names', 'pvals']})
    df.to_csv(cluster_marker_genes_fp, sep="\t", index=False)

def plot_annotated_cell_types(args, adatas, annotations_list, dataset="chicken", scatter_sz= 4):
    output_dir = f'{args.output_dir}/{dataset}/merged'
    mkdir(output_dir)
    samples = ['D4', 'D7', 'D10', 'D14']
    fig, axs = figure(1, len(samples), rsz=2.8, csz=5.3, wspace=.5, hspace=.4)
    annotations_names, annotation_colors = get_cell_type_annotation_colors()
    for sid, sample in enumerate(samples):
        ax = axs[sid]
        ax.axis('off')
        coord = adatas[sid].obsm['spatial'].astype(float)
        x, y = coord[:, 0], coord[:, 1]
        annotations = annotations_list[sid]
        uniq_annots = np.unique(annotations)
        box_width_ratio = .8 if sample != "D14" else .6
        for cid, annot in enumerate(uniq_annots):
            ind = annotations == annot
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=annotation_colors[annot], label=annot)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * box_width_ratio, box.height])
        lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5))
        for handle in lgnd.legendHandles:
            handle._sizes = [10]
        ax.set_title(sample, fontsize=title_sz)
    fig_fp = f"{output_dir}/annotated_cell_types.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_annotated_cell_regions(args, adatas, annotations_list, dataset="chicken", scatter_sz= 4):
    output_dir = f'{args.output_dir}/{dataset}/merged'
    mkdir(output_dir)
    samples = ['D4', 'D7', 'D10', 'D14']
    fig, axs = figure(1, len(samples), rsz=2.8, csz=5.3, wspace=.5, hspace=.4)
    annotations_names, annotation_colors = get_region_annotation_colors()
    for sid, sample in enumerate(samples):
        ax = axs[sid]
        ax.axis('off')
        coord = adatas[sid].obsm['spatial'].astype(float)
        x, y = coord[:, 0], coord[:, 1]
        annotations = annotations_list[sid]
        uniq_annots = np.unique(annotations)
        box_width_ratio = .8 if sample != "D14" else .6
        for cid, annot in enumerate(uniq_annots):
            ind = annotations == annot
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=annotation_colors[annot], label=annot)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * box_width_ratio, box.height])
        lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5))
        for handle in lgnd.legendHandles:
            handle._sizes = [10]
        ax.set_title(sample, fontsize=title_sz)
    fig_fp = f"{output_dir}/annotated_regions.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_annotated_clusters(args, adatas, sample_name="merged", dataset="chicken", method="leiden", scatter_sz= 4):
    args.spatial = True
    output_dir = f'{args.output_dir}/{dataset}/{sample_name}'
    mkdir(output_dir)
    samples = ['D4', 'D7', 'D10', 'D14']
    fig, axs = figure(1, len(samples), rsz=2.8, csz=5.3, wspace=.5, hspace=.4)
    annotation_colors = get_cluster_colors(args, samples)
    for sid, sample in enumerate(samples):
        ax = axs[sid]
        ax.axis('off')
        coord = adatas[sid].obsm['spatial'].astype(float)
        x, y = coord[:, 0], coord[:, 1]
        box_width_ratio = .8 if sample != "D14" else .6
        cluster_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample)}'
        pred_clusters = pd.read_csv(f"{cluster_dir}/{method}.tsv", header=None).values.flatten().astype(int)
        uniq_pred = np.unique(pred_clusters)
        uniq_pred = sorted(uniq_pred, key=lambda cluster: annotation_colors[sid][cluster][0])
        for cid, cluster in enumerate(uniq_pred):
            ind = pred_clusters == cluster
            label, color = annotation_colors[sid][cluster]
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=f"#{color}", label=label)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * box_width_ratio, box.height])
        lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5))
        for handle in lgnd.legendHandles:
            handle._sizes = [10]
        ax.set_title(sample, fontsize=title_sz)
    fig_fp = f"{output_dir}/annotated_clusters.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_expr_in_ST(args, adatas, gene_name, sample_name = "merged", dataset="chicken", scatter_sz= 6, cm = plt.get_cmap("magma")):
    args.spatial = True
    output_dir = f'{args.output_dir}/{dataset}/{sample_name}/expr_in_ST'
    mkdir(output_dir)
    samples = ['D4', 'D7', 'D10', 'D14']
    fig, axs = figure(1, len(samples), rsz=2.8, csz=4.8, wspace=.5, hspace=.4)
    exprs = [np.asarray(adatas[sid][:, adatas[sid].var_names == gene_name].X.todense()).flatten() for sid, sample in enumerate(samples)]
    box_ratios = [.9, .9, 1., .8]
    for sid, sample in enumerate(samples):
        ax = axs[sid]
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        adata = adatas[sid]
        coord = adata.obsm['spatial'].astype(float)
        x, y = coord[:, 0], coord[:, 1]
        expr = exprs[sid]
        st = ax.scatter(x, y, s=scatter_sz, c=expr, cmap=cm)#, vmin=0, vmax=40
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * box_ratios[sid], box.height])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = fig.colorbar(st, cax=cax)
        clb.ax.set_ylabel("Expr", labelpad=10, rotation=270, fontsize=10, weight='bold')
        ax.set_title(sample, fontsize=title_sz)
        if sid == 0:
            ax.set_ylabel(gene_name, fontsize=title_sz)
    fig_fp = f"{output_dir}/{gene_name}_ST_expr.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_lineage_tsne(args, adata, cell_types, regions, days, cluster_annotations, lineage, dataset="chicken", scatter_sz= 2, n_neighbors=8, embedding=True):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, lineage)}'
    #lineage_clusters, lineage_cluster_color_dict = get_lineage_clusters_and_color_dict(args, lineage)

    mkdir(output_dir)
    annotation_types = ["Cluster annotation", "Embryonic Day", "Inferred cell type", "Inferred region", "Lineage Cluster"]
    fig, axs = figure(1, len(annotation_types), rsz=2.8, csz=6.6, wspace=.5, hspace=.4)
    combined_cluster = get_combined_merged_annotations(days, cluster_annotations)
    annotations_list = [cluster_annotations, days, cell_types, regions, combined_cluster]

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.tsne(adata, use_rep='X')
    tsne_positions = adata.obsm["X_tsne"]
    for aid, annotation_type in enumerate(annotation_types):
        print(f"Processing annotation type: {annotation_type}")
        ax = axs[aid]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t-SNE 1", fontsize=12, color="black")
        ax.set_ylabel("t-SNE 2", fontsize=12, color="black")
        annotations = annotations_list[aid]
        uniq_annot = np.unique(annotations)

        if aid == 0:
            color_dict = get_cluster_colors_dict(args)
        elif aid == 1:
            color_dict = get_day_colors()
        elif aid == 2:
            _, color_dict = get_cell_type_annotation_colors()
        elif aid == 3:
            _, color_dict = get_region_annotation_colors()
        else:
            # color_dict = lineage_cluster_color_dict
            ncluster = uniq_annot.size
            cm = plt.get_cmap("tab20")
            color_dict = {annot: cm((cid * (ncluster / (ncluster - 1.0))) / ncluster) for cid, annot in
                          enumerate(uniq_annot)}

        for cid, annot in enumerate(uniq_annot):
            tsne_sub = tsne_positions[annotations == annot]
            ax.scatter(tsne_sub[:, 0], tsne_sub[:, 1], s=scatter_sz, color=color_dict[annot], label=annot)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5))
        for handle in lgnd.legendHandles:
            handle._sizes = [10]
    name = "embedding" if embedding else "expr"
    fig_fp = f"{output_dir}/lineage_{lineage}_{name}_tsne.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_lineage_expr_tsne(args, adata, cell_types, regions, days, cluster_annotations, lineage, dataset="chicken", scatter_sz= 2):
    plot_lineage_tsne(args, adata, cell_types, regions, days, cluster_annotations, lineage, scatter_sz= scatter_sz, embedding=False)

def plot_lineage_embedding_tsne(args, adata, cell_types, regions, days, cluster_annotations, lineage, dataset="chicken", scatter_sz= 2):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, lineage)}'
    feature_fp = os.path.join(output_dir, "features.tsv")
    adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
    plot_lineage_tsne(args, adata, cell_types, regions, days, cluster_annotations, lineage, scatter_sz= scatter_sz, embedding=True)

def plot_lineage_umap(args, adata, cell_types, regions, days, cluster_annotations, lineage, dataset="chicken", scatter_sz= 2, n_neighbors=8, embedding=True):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, lineage)}'
    #lineage_clusters, lineage_cluster_color_dict = get_lineage_clusters_and_color_dict(args, lineage)

    mkdir(output_dir)
    annotation_types = ["Cluster annotation", "Embryonic Day", "Inferred cell type", "Inferred region", "Lineage Cluster"]
    fig, axs = figure(1, len(annotation_types), rsz=2.8, csz=6.6, wspace=.5, hspace=.4)
    combined_cluster = get_combined_merged_annotations(days, cluster_annotations)
    annotations_list = [cluster_annotations, days, cell_types, regions, combined_cluster]

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.leiden(adata, resolution=.8)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)
    sc.tl.umap(adata, init_pos="paga")
    umap_positions = adata.obsm["X_umap"]
    for aid, annotation_type in enumerate(annotation_types):
        print(f"Processing annotation type: {annotation_type}")
        ax = axs[aid]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1", fontsize=12, color="black")
        ax.set_ylabel("UMAP 2", fontsize=12, color="black")

        annotations = annotations_list[aid]
        uniq_annot = np.unique(annotations)
        if aid == 0:
            color_dict = get_cluster_colors_dict(args)
        elif aid == 1:
            color_dict = get_day_colors()
        elif aid == 2:
            _, color_dict = get_cell_type_annotation_colors()
        elif aid == 3:
            _, color_dict = get_region_annotation_colors()
        else:
            # color_dict = lineage_cluster_color_dict
            ncluster = uniq_annot.size
            cm = plt.get_cmap("tab20")
            color_dict = {annot: cm((cid * (ncluster / (ncluster - 1.0))) / ncluster) for cid, annot in
                          enumerate(uniq_annot)}

        for cid, annot in enumerate(uniq_annot):
            umap_sub = umap_positions[annotations == annot]
            ax.scatter(umap_sub[:, 0], umap_sub[:, 1], s=scatter_sz, color=color_dict[annot], label=annot)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5))
        for handle in lgnd.legendHandles:
            handle._sizes = [10]

    name = "embedding" if embedding else "expr"
    fig_fp = f"{output_dir}/lineage_{lineage}_{name}_umap.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')


def plot_lineage_expr_umap(args, adata, cell_types, regions, days, cluster_annotations, lineage, dataset="chicken", scatter_sz= 2):
    plot_lineage_umap(args, adata, cell_types, regions, days, cluster_annotations, lineage, scatter_sz= scatter_sz, embedding=False)

def plot_lineage_embedding_umap(args, adata, cell_types, regions, days, cluster_annotations, lineage, dataset="chicken", scatter_sz= 2):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, lineage)}'
    feature_fp = os.path.join(output_dir, "features.tsv")
    adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
    plot_lineage_umap(args, adata, cell_types, regions, days, cluster_annotations, lineage, scatter_sz= scatter_sz, embedding=True)


def plot_lineage_phate(args, data_phate, cell_types, regions, days, cluster_annotations, lineage, dataset="chicken", scatter_sz= 4, embedding=True):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, lineage)}'
    mkdir(output_dir)
    #lineage_clusters, lineage_cluster_color_dict = get_lineage_clusters_and_color_dict(args, lineage)
    annotation_types = ["Cluster annotation", "Embryonic Day", "Inferred cell type", "Inferred region",
                        "Lineage Cluster"]
    fig, axs = figure(1, len(annotation_types), rsz=2.8, csz=6.6, wspace=.5, hspace=.4)
    combined_cluster = get_combined_merged_annotations(days, cluster_annotations)
    annotations_list = [cluster_annotations, days, cell_types, regions, combined_cluster]
    for aid, annotation_type in enumerate(annotation_types):
        print(f"Processing annotation type: {annotation_type}")
        ax = axs[aid]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("PHATE 1", fontsize=12, color="black")
        ax.set_ylabel("PHATE 2", fontsize=12, color="black")

        if aid == 0:
            color_dict = get_cluster_colors_dict(args)
        elif aid == 1:
            color_dict = get_day_colors()
        elif aid == 2:
            _, color_dict = get_cell_type_annotation_colors()
        elif aid == 3:
            _, color_dict = get_region_annotation_colors()
        else:
            # color_dict = lineage_cluster_color_dict
            ncluster = uniq_annot.size
            cm = plt.get_cmap("tab20")
            color_dict = {annot: cm((cid * (ncluster / (ncluster - 1.0))) / ncluster) for cid, annot in
                          enumerate(uniq_annot)}
        annotations = annotations_list[aid]

        uniq_annot = np.unique(annotations)

        for cid, annot in enumerate(uniq_annot):
            phate_sub = data_phate[annotations == annot]
            ax.scatter(phate_sub[:, 0], phate_sub[:, 1], s=scatter_sz, color=color_dict[annot], label=annot)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5))
        for handle in lgnd.legendHandles:
            handle._sizes = [10]
    name = "embedding" if embedding else "expr"
    fig_fp = f"{output_dir}/lineage_{lineage}_{name}_phate.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_lineage_embedding_phate(args, cell_types, regions, days, cluster_annotations, lineage, dataset="chicken", scatter_sz= 2):
    phate_op, data_phate = get_phate(args, lineage)
    plot_lineage_phate(args, data_phate, cell_types, regions, days, cluster_annotations, lineage, embedding=True, scatter_sz= scatter_sz)

def plot_lineage_expr_phate(args, adata, cell_types, regions, days, cluster_annotations, lineage, dataset="chicken", scatter_sz= 2):
    phate_op, data_phate = get_phate_by_adata(adata)
    plot_lineage_phate(args, data_phate, cell_types, regions, days, cluster_annotations, lineage, embedding=False, scatter_sz= scatter_sz)

def plot_lineage_annotated_clusters(args, adatas, lineage_adatas, lineage_name, dataset="chicken", method="leiden", scatter_sz= 4, cm = plt.get_cmap("Spectral")):
    args.spatial = True
    output_dir = f'{args.output_dir}/{dataset}/{lineage_name}'
    mkdir(output_dir)
    samples = ['D4', 'D7', 'D10', 'D14']
    fig, axs = figure(1, len(samples), rsz=2.8, csz=5.3, wspace=.5, hspace=.4)
    cluster_dir = f'{args.output_dir}/{get_target_fp(args, dataset, lineage_name)}'
    pred_clusters = pd.read_csv(f"{cluster_dir}/{method}.tsv", header=None).values.flatten().astype(int)
    num_samples = [0]
    for ni in range(len(lineage_adatas)-1):
        num_samples.append(num_samples[-1] + lineage_adatas[ni].n_obs)
    clusters = np.split(pred_clusters, num_samples[1:])
    uniq_pred = np.unique(pred_clusters)
    ncluster = uniq_pred.size
    lineages = lineage_name.split("_")
    for sid, sample in enumerate(samples):
        ax = axs[sid]
        ax.axis('off')
        coord = adatas[sid].obsm['spatial'].astype(float)
        x, y = coord[:, 0], coord[:, 1]
        ax.scatter(x, y, s=scatter_sz, color="#adb5bd")
        for lid, lineage in enumerate(lineages):
            offset = lid * len(samples) + sid
            if lineage_adatas[offset].n_obs:
                lc = lineage_adatas[offset].obsm['spatial'].astype(float)
                x, y = lc[:, 0], lc[:, 1]

                for cid, cluster in enumerate(uniq_pred):
                    ind = clusters[offset] == cluster
                    color = cm((cid * (ncluster / (ncluster - 1.0))) / ncluster)
                    ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster)

        box_width_ratio = .8 if sample != "D14" else .6

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * box_width_ratio, box.height])
        lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5))
        for handle in lgnd.legendHandles:
            handle._sizes = [10]
        ax.set_title(sample, fontsize=title_sz)
    fig_fp = f"{output_dir}/annotated_clusters.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_phate_pseudotime(args, lineage, lineage_adata, days, dataset="chicken", scatter_sz= 2, cm = plt.get_cmap("gist_rainbow")):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, lineage)}'
    fig, ax = figure(1, 1, rsz=2.8, csz=2.8, wspace=.5, hspace=.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PHATE 1", fontsize=12, color="black")
    ax.set_ylabel("PHATE 2", fontsize=12, color="black")

    phate_op, data_phate = get_phate(args, lineage)
    pca = PCA(n_components=1)
    phate_diffusion_comp_pca = pca.fit_transform(phate_op.diff_op).flatten()
    max_pca_comp, min_pca_comp = np.max(phate_diffusion_comp_pca), np.min(phate_diffusion_comp_pca)
    phate_diffusion_comp_pca = (phate_diffusion_comp_pca - min_pca_comp)/(max_pca_comp - min_pca_comp)
    max_extreme_index, min_extreme_index = np.argmax(data_phate[:, -1]), np.argmin(phate_diffusion_comp_pca)
    root_cell_index = max_extreme_index#root_cell_index = min_extreme_index if days[min_extreme_index] == 'D4' else max_extreme_index

    pseudotimes = get_pseudotime(lineage_adata)
    # pseudotimes = phate_diffusion_comp_pca if root_cell_index == min_extreme_index else 1 - phate_diffusion_comp_pca
    st = ax.scatter(data_phate[:, 0], data_phate[:, 1], s=scatter_sz, c=pseudotimes, cmap=cm)
    # ax.scatter(data_phate[:, 0], data_phate[:, 1], s=scatter_sz, color="grey",
    #            cmap=cm)
    ax.scatter(data_phate[max_extreme_index, 0], data_phate[max_extreme_index, 1], s=scatter_sz, color="black")#, c=pseudotimes
    ax.scatter(data_phate[min_extreme_index, 0], data_phate[min_extreme_index, 1], s=scatter_sz, color="grey")  # , c=pseudotimes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = fig.colorbar(st, cax=cax)
    clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    fig_fp = f"{output_dir}/{lineage}_phate_pseudotime.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def get_pseudotime(adata, resolution = .8, n_neighbors = 10):
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution)
    sc.tl.paga(adata)
    expr = adata.obsm["X_pca"]
    distances = distance_matrix(expr, expr)
    sum_dists = distances.sum(axis=1)
    adata.uns['iroot'] = np.argmax(sum_dists)
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)
    pseudotimes = adata.obs['dpt_pseudotime'].to_numpy()
    return pseudotimes

def isAnnotationInLineage(anno, lineages):
    for lineage in lineages:
        if anno.startswith(lineage):
            return True
    return False

def plot_lineage_pseudotime(args, adatas, annotations_list, lineage, lineage_adata, days, dataset="chicken", scatter_sz= 4, cm = plt.get_cmap("gist_rainbow")):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, lineage)}'
    mkdir(output_dir)
    samples = ['D4', 'D7', 'D10', 'D14']
    fig, axs = figure(1, len(samples), rsz=2.8, csz=5.3, wspace=.5, hspace=.4)
    # phate_op, data_phate = get_phate(args, lineage)
    # pca = PCA(n_components=1)
    # phate_diffusion_comp_pca = pca.fit_transform(phate_op.diff_op).flatten()
    # max_extreme_index, min_extreme_index = np.argmax(data_phate[:, -1]), np.argmin(phate_diffusion_comp_pca)
    # root_cell_index = max_extreme_index#min_extreme_index if days[min_extreme_index] == 'D4' else max_extreme_index
    # args.spatial = True
    # feature_dir = f'{args.output_dir}/{get_target_fp(args, dataset, lineage)}'
    # feature_fp = os.path.join(feature_dir, "features.tsv")
    # adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
    pseudotimes = get_pseudotime(lineage_adata)

    for sid, sample in enumerate(samples):
        ax = axs[sid]
        ax.axis('off')
        ind = np.array([True if not isAnnotationInLineage(annotation, lineage) else False for annotation in annotations_list[sid]])
        coord = adatas[sid][ind, :].obsm['spatial'].astype(float)
        x, y = coord[:, 0], coord[:, 1]
        ax.scatter(x, y, s=scatter_sz, color="#adb5bd")

        coord = adatas[sid][~ind, :].obsm['spatial'].astype(float)
        x, y = coord[:, 0], coord[:, 1]
        pseudotimes_sub = pseudotimes[days == sample]
        st = ax.scatter(x, y, s=scatter_sz, c=pseudotimes_sub, cmap=cm)
        box_width_ratio = "5%" if sample != "D14" else "20%"
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=box_width_ratio, pad=0.05)
        clb = fig.colorbar(st, cax=cax)
        clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
        ax.set_title(sample, fontsize=title_sz)
    fig_fp = f"{output_dir}/{lineage}_ST_pseudotime.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

####################################
#-------------Pipelines------------#
####################################

def basic_pipeline(args):
    sample_list = ['D4', 'D7', 'D10', 'D14']

    for sample_name in sample_list:
        cell_types, region_annos = get_annotations_chicken(args, sample_name)
        adata = load_chicken_data(args, sample_name)
        train_pipeline(args, adata, sample_name, cell_types)
        # plot_clustering(args, sample_name)
        # plot_pseudotime(args, sample_name)
        # plot_phate(args, sample_name, adata_filtered.X, anno_clusters)
        #adata_filtered.obs["bulk_labels"] = pd.Categorical(anno_clusters)
        pred_clusters = get_clusters(args, sample_name)
        plot_rank_marker_genes_group(args, sample_name, adata_filtered, pred_clusters)

def annotation_pipeline(args):
    sample_list = ['D4', 'D7', 'D10', 'D14']
    adatas, cell_type_annotations_list, region_annotations_list = [], [], []
    for sample_name in sample_list:
        anno_clusters, region_annos = get_annotations_chicken(args, sample_name)
        cell_type_annotations_list.append(anno_clusters)
        region_annotations_list.append(region_annos)
        adata = load_chicken_data(args, sample_name)
        adatas.append(adata)
    # plot_annotated_cell_types(args, adatas, cell_type_annotations_list)
    # plot_annotated_cell_regions(args, adatas, region_annotations_list)
    plot_annotated_clusters(args, adatas)

def train_pipeline(args, adata, sample_name, cell_types, dataset="chicken", clustering_method="leiden", resolution = .8, n_neighbors = 10, isTrain=True):
    adata_filtered, expr, genes, cells, spatial_graph, spatial_dists = preprocessing_data(args, adata)
    if isTrain:
        for spatial in [True]:
            args.spatial = spatial
            embedding = train(args, expr, spatial_graph, spatial_dists)
            save_features(args, embedding, dataset, sample_name)
            clustering(args, dataset, sample_name, clustering_method, n_neighbors=n_neighbors, resolution=resolution)
            # pseudotime(args, dataset, sample_name, root_cell_type="Epi-epithelial cells", cell_types=cell_types, n_neighbors=n_neighbors, resolution=resolution)
    return adata_filtered, genes, cells

def hiearchical_clustering_heatmap(args, data, lineage_name, annotation_types, annotations_arr, annotation_color_dict_arr, annotation_colors_arr, dataset="chicken"):
    output_dir = f'{args.output_dir}/{dataset}/{lineage_name}'
    mkdir(output_dir)
    fig_fp = f"{output_dir}/h_clustering_heatmap.pdf"
    print("Hiearchical clustering..")
    locs = [[0.5, 0.0], [0.5, 0.35], [0.5, 0.6]]
    g = sns.clustermap(data, metric="correlation", cmap='viridis', row_colors=annotation_colors_arr, linewidths=0, xticklabels=False, yticklabels=False,
                       dendrogram_ratio=(.1, .4))
    g.ax_row_dendrogram.set_visible(False)
    g.cax.set_visible(False)
    for aid, annotations in enumerate(annotations_arr):
        for label in np.unique(annotations):
            g.ax_row_dendrogram.bar(0, 0, color=annotation_color_dict_arr[aid][label], label=label, linewidth=0)
        g.ax_row_dendrogram.legend(title=annotation_types[aid], loc=f'center right', ncol=1, bbox_to_anchor=locs[aid])
    g.savefig(fig_fp, dpi=300)

def get_annotation_color_arr(args, cluster_annotations, days, regions):
    cluster_annotations_color_dict = get_cluster_colors_dict(args)
    cluster_colors = [cluster_annotations_color_dict[anno] for anno in cluster_annotations]

    days_color_dict = get_day_colors()
    days_colors = [days_color_dict[anno] for anno in days]

    # _, cell_type_color_dict = get_cell_type_annotation_colors()
    # cell_type_colors = [cell_type_color_dict[anno] for anno in cell_types]
    _, region_color_dict = get_region_annotation_colors()
    region_colors = [region_color_dict[anno] for anno in regions]
    return [cluster_colors, days_colors, region_colors], [cluster_annotations_color_dict, days_color_dict, region_color_dict]

def heatmap_pipeline(args, all_lineage=True):
    sample_list = ['D7', 'D10', 'D14']
    adatas, cell_types, regions, days = [], [], [], []
    for sample_name in sample_list:
        adata = load_chicken_data(args, sample_name)
        adatas.append(adata)

        sample_cell_types, sample_regions = get_annotations_chicken(args, sample_name)
        cell_types.append(sample_cell_types)
        regions.append(sample_regions)

        sample_days = np.array([sample_name for _ in range(sample_cell_types.shape[0])])
        days.append(sample_days)
    cluster_annotations = get_cluster_annotations_chicken(args, sample_list)
    if all_lineage:
        lineage_name = "merged"
        merged_adata, merged_cluster_annotations, merged_cell_types, merged_regions, merged_days = merge_adatas_annotations_chicken(
            adatas, cluster_annotations, cell_types, regions, days)
    else:
        lineages = ["Valve cells", "MT-enriched valve cells"]
        lineage_name = "_".join(lineages)
        filtered_adatas, filtered_annotations, filtered_cell_types, filtered_regions, filtered_days = filter_adatas_annotations_chicken(
            adatas, cluster_annotations, cell_types, regions, days, lineages)
        merged_adata, merged_cluster_annotations, merged_cell_types, merged_regions, merged_days = merge_adatas_annotations_chicken(
            filtered_adatas, filtered_annotations, filtered_cell_types, filtered_regions, filtered_days)
    annotation_colors_arr, color_dict_arr = get_annotation_color_arr(args, merged_cluster_annotations, merged_days, merged_regions)
    adata_filtered, expr, genes, cells, spatial_graph, spatial_dists = preprocessing_data(args, merged_adata, n_top_genes=101)
    expr = adata_filtered.X.todense()
    distances = cdist(expr, expr, 'cosine')
    annotation_types = ["Cluster", "Day", "Region"]
    hiearchical_clustering_heatmap(args, distances, lineage_name, annotation_types, [merged_cluster_annotations, merged_days, merged_regions], color_dict_arr, annotation_colors_arr)

    method = "combined_cluster"
    top_n_cluster_specific_genes = get_top_n_cluster_specific_genes(args, lineage_name, method=method, top_n=3)
    annotation_types = ["Cluster", "Day", "Region"]
    adata_filtered = adata_filtered[:,
                     [True if g in top_n_cluster_specific_genes else False for g in genes]].X.todense()
    annotation_colors_arr, color_dict_arr = get_annotation_color_arr(args, merged_cluster_annotations, merged_days,
                                                                     merged_regions)
    hiearchical_clustering_heatmap(args, adata_filtered, lineage_name, annotation_types,
                                   [merged_cluster_annotations, merged_days, merged_regions], color_dict_arr,
                                   annotation_colors_arr)

def get_combined_annotations(days, clusters):
    return np.array([f"{day}-{clusters[did]}"for did, day in enumerate(days)])

def get_combined_merged_annotations(days, clusters):
    return np.array(["%s-%s" % (day, clusters[did].split('-')[0]) for did, day in enumerate(days)])

def get_top_n_cluster_specific_genes(args, sample_name, method, dataset="chicken", top_n=3):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    cluster_marker_genes_fp = f'{output_dir}/marker_genes_pval_gby_{method}.tsv'
    df = pd.read_csv(cluster_marker_genes_fp, sep="\t")
    genes = df.loc[:top_n, df.columns.str.endswith("_n")].values.flatten().astype(str)
    return np.unique(genes)

def lineage_pipeline(args, all_lineage=False):
    sample_list = ['D4', 'D7', 'D10', 'D14']#
    adatas, cell_types, regions, days = [], [], [], []
    for sample_name in sample_list:
        adata = load_chicken_data(args, sample_name)
        adatas.append(adata)

        sample_cell_types, sample_regions = get_annotations_chicken(args, sample_name)
        cell_types.append(sample_cell_types)
        regions.append(sample_regions)

        sample_days = np.array([sample_name for _ in range(sample_cell_types.shape[0])])
        days.append(sample_days)

    cluster_annotations = get_cluster_annotations_chicken(args, sample_list)
    if all_lineage:
        lineage_name = "merged"
        merged_adata, merged_cluster_annotations, merged_cell_types, merged_regions, merged_days = merge_adatas_annotations_chicken(adatas, cluster_annotations, cell_types, regions, days)
    else:
        lineages = ["Valve cells", "MT-enriched valve cells"]
        lineage_name = "_".join(lineages)
        filtered_adatas, filtered_annotations, filtered_cell_types, filtered_regions, filtered_days = filter_adatas_annotations_chicken(adatas, cluster_annotations, cell_types, regions, days, lineages)
        merged_adata, merged_cluster_annotations, merged_cell_types, merged_regions, merged_days = merge_adatas_annotations_chicken(filtered_adatas, filtered_annotations, filtered_cell_types, filtered_regions, filtered_days)

    adata_filtered, genes, cells = train_pipeline(args, merged_adata, lineage_name, merged_cell_types, resolution=.4, isTrain=False)
    # plot_lineage_annotated_clusters(args, adatas, filtered_adatas, lineage_name)
    #plot_lineage_embedding_tsne(args, adata_filtered, merged_cell_types, merged_regions, merged_days, merged_cluster_annotations, lineage_name)
    # plot_lineage_expr_umap(args, adata_filtered, merged_cell_types, merged_regions, merged_days, merged_cluster_annotations, lineage_name, scatter_sz= 1)
    plot_lineage_embedding_umap(args, adata_filtered, merged_cell_types, merged_regions, merged_days, merged_cluster_annotations, lineage_name)
    # plot_lineage_expr_phate(args, adata_filtered, merged_cell_types, merged_regions, merged_days, merged_cluster_annotations, lineage_name)
    #plot_lineage_embedding_phate(args, merged_cell_types, merged_regions, merged_days, merged_cluster_annotations, lineage_name)
    # pred_clusters = get_clusters(args, lineage_name)
    #combined_annotations = get_combined_annotations(merged_days, merged_cluster_annotations)

    #plot_lineage_expr_tsne(args, adata_filtered, merged_cell_types, merged_regions, merged_days, combined_merged_annotations, lineage_name, scatter_sz= 1)
    #method = "combined_merged_cluster"
    #method = "clustering"
    #plot_rank_marker_genes_group(args, lineage_name, adata_filtered, combined_merged_annotations, method=method)

    #plot_rank_marker_genes_group(args, lineage_name, merged_adata, merged_days, method="days")
    # plot_phate_pseudotime(args, lineage, merged_adata, merged_days)
    # plot_lineage_pseudotime(args, adatas, cluster_annotations, lineage_name, adata_filtered, merged_days)

def expr_analysis_pipeline(args):
    sample_list = ['D4', 'D7', 'D10', 'D14']
    adatas = [load_chicken_data(args, sample) for sample in sample_list]
    target_genes = ["CNMD", "BAMBI", "COL1A1", "S100A6", "S100A11", "TXNDC5"]#["ACADSB", "ACBD7", "ACTA2", "ACTG2", "AKR1D1", "APOA1", "APP", "ATP6V1E1", "BAMBI", "BMP10", "BRD2", "C1H2ORF40", "C5H11orf58", "CA9", "CAV3", "CCDC80", "CD36", "CHGB", "CHODL", "CIAO2B", "CNMD", "COL14A1", "COL1A1", "COL4A1", "COL5A1", "COX17", "CPE", "CRIP1", "CSRP2", "CSTA", "CTGF", "CTSA", "DERA", "DPYSL3", "DRAXIN", "EDNRA", "ENSGALG00000004518", "ENSGALG00000013239", "ENSGALG00000015349", "ENSGALG00000020788", "ENSGALG00000028551", "ENSGALG00000040263", "ENSGALG00000050984", "ENSGALG00000053871", "FABP3", "FABP5", "FABP7", "FBLN1", "FGFR3", "FHL1", "FHL2", "FMC1", "FSTL1", "FXYD6", "GJA5", "GKN2", "GLRX5", "GPC1", "GPX3", "HADHB", "HAPLN3", "HBBR-1", "HPGD", "ID2", "ID4", "IRX4", "KRT18", "LBH", "LDHA", "LMOD2", "LSP1", "LTBP2", "LUM", "MAD2L2", "MAPK6", "MAPRE1", "MB"] + ["MFAP2", "MGP", "MOXD1", "MSX1", "MT4L", "MTFP1", "MUSTN1", "MYH1D", "MYH1F", "MYH7", "MYL1", "MYLK", "MYOM1", "MYOM2", "MYOZ2", "NIPSNAP2", "NPC2", "NRN1L", "OSTN", "OXCT1", "PECAM1", "PENK", "PERP2", "PGAP2", "PITX2", "PLN", "POSTN", "PRNP", "PRRX1", "RAMP2", "RARRES1", "RCSD1", "RD3L", "RRAD", "RSRP1", "S100A11", "S100A6", "SEC63", "SERPINE2", "SESTD1", "SFRP1", "SFRP2", "SLN", "SMAD6", "SYPL1", "TBX5", "TESC", "TFPI2", "THBS4", "TIMM9", "TMEM158", "TMEM163", "TNIP1", "TNNC2", "TPM1", "TUBAL3", "TXNDC5", "VCAN", "Wpkci-7"]#"BAMBI"
    for gene in target_genes:
        plot_expr_in_ST(args, adatas, gene)






