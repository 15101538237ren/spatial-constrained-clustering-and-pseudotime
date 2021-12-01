# -*- coding:utf-8 -*-
import phate
import anndata
import warnings
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

def filter_adatas_annotations_chicken(adatas, cluster_annotations, cell_types, regions, days, lineage):

    filtered_adatas, filtered_annotations, filtered_cell_types, filtered_regions, filtered_days = [], [], [], [], []

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

def rank_marker_genes_group(args, sample_name, adata_filtered, method="leiden", dataset="chicken"):
    original_spatial = args.spatial
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(str)
    adata_filtered.obs["leiden"] = pd.Categorical(pred_clusters)
    sc.tl.rank_genes_groups(adata_filtered, 'leiden', method='wilcoxon')
    sc.pl.rank_genes_groups(adata_filtered, n_genes=25, ncols=5, fontsize=10, sharey=False, save=f"{sample_name}_ranks.pdf")
    sc.pl.rank_genes_groups_heatmap(adata_filtered, n_genes=3, standard_scale='var', save=f"{sample_name}_heatmap.pdf")
    sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=3, standard_scale='var', save=f"{sample_name}_mean_expr.pdf")
    sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=3, values_to_plot="logfoldchanges", cmap='bwr', vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change', save=f"{sample_name}_dot_lfc.pdf")
    sc.pl.rank_genes_groups_matrixplot(adata_filtered, n_genes=3, values_to_plot="logfoldchanges", cmap='bwr', vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change', save=f"{sample_name}_matrix_lfc.pdf")
    args.spatial = original_spatial
    cluster_marker_genes_fp = f'{output_dir}/marker_genes_pval.tsv'
    result = adata_filtered.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    df = pd.DataFrame(
        {group + '_' + key[:1]: result[key][group]
        for group in groups for key in ['names', 'pvals']})
    df.to_csv(cluster_marker_genes_fp, sep="\t", index=False)

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

def plot_annotated_clusters(args, adatas, dataset="chicken", method="leiden", scatter_sz= 4):
    args.spatial = True
    output_dir = f'{args.output_dir}/{dataset}/merged'
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
        # rank_marker_genes_group(args, sample_name, adata_filtered)

def annotation_pipeline(args):
    sample_list = ['D4', 'D7', 'D10', 'D14']
    adatas, cell_type_annotations_list, region_annotations_list = [], [], []
    for sample_name in sample_list:
        anno_clusters, region_annos = get_annotations_chicken(args, sample_name)
        cell_type_annotations_list.append(anno_clusters)
        region_annotations_list.append(region_annos)
        adata = load_chicken_data(args, sample_name)
        adatas.append(adata)
    plot_annotated_cell_types(args, adatas, cell_type_annotations_list)
    plot_annotated_cell_regions(args, adatas, region_annotations_list)
    plot_annotated_clusters(args, adatas)

def train_pipeline(args, adata, sample_name, cell_types, dataset="chicken", clustering_method="leiden", resolution = .8, n_neighbors = 10):
    adata_filtered, expr, genes, cells, spatial_graph, spatial_dists = preprocessing_data(args, adata)
    for spatial in [False, True]:
        args.spatial = spatial
        embedding = train(args, expr, spatial_graph, spatial_dists)
        save_features(args, embedding, dataset, sample_name)
        clustering(args, dataset, sample_name, clustering_method, n_neighbors=n_neighbors, resolution=resolution)
        # pseudotime(args, dataset, sample_name, root_cell_type="Epi-epithelial cells", cell_types=cell_types, n_neighbors=n_neighbors, resolution=resolution)

def lineage_pipeline(args, lineage="Vascular smooth muscle cells"):
    sample_list = ['D4', 'D7', 'D10', 'D14']
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

    filtered_adatas, filtered_annotations, filtered_cell_types, filtered_regions, filtered_days = filter_adatas_annotations_chicken(adatas, cluster_annotations, cell_types, regions, days, lineage)
    merged_adata, merged_cluster_annotations, merged_cell_types, merged_regions, merged_days = merge_adatas_annotations_chicken(filtered_adatas, filtered_annotations, filtered_cell_types, filtered_regions, filtered_days)
    train_pipeline(args, merged_adata, lineage, merged_cell_types)




