# -*- coding:utf-8 -*-
import math
import phate
import anndata
import shutil
import warnings
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon, pearsonr
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

def get_adata_from_embeddings(args, sample_name, dataset="breast_cancer"):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    feature_fp = os.path.join(output_dir, "features.tsv")
    adata = sc.read_csv(feature_fp, delimiter="\t", first_column_names=None)
    return adata

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
    df = df.loc[:top_n-1, df.columns.str.endswith("_n")]
    cluster_specific_genes_dict = {}
    for cluster_abbr in df.columns:
        cluster_specific_genes_dict[cluster_abbr.strip("_n")] = df[cluster_abbr].values.astype(str)
    return cluster_specific_genes_dict

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


def get_GO_term_dict(args):
    base_dir = f"{args.dataset_dir}/Visium/Breast_Cancer/analysis"
    genes_with_go_ids_fp = f'{base_dir}/genes_with_go_ids.csv'
    go_id_to_genes_dict_pkl_fp = f"{base_dir}/go_id_to_genes_dict.pkl"

    if os.path.exists(go_id_to_genes_dict_pkl_fp):
        with open(go_id_to_genes_dict_pkl_fp, 'rb') as f:
            go_terms_dict = pickle.load(f)
            return go_terms_dict
    else:
        df = pd.read_csv(genes_with_go_ids_fp).values.astype(str)
        go_terms = np.array(np.unique(df[:, 1]))
        go_terms_dict = {go_id : df[df[:, 1] == go_id, 0] for go_id in go_terms}

        with open(go_id_to_genes_dict_pkl_fp, 'wb') as f:
            pickle.dump(go_terms_dict, f, -1)
            print(f"Saved at {go_id_to_genes_dict_pkl_fp}")

        return go_terms_dict

def get_GO_terms_with_spatial_coherent_expr(args, adata, sample_name, go_term_dict, dataset="breast_cancer"):
    coords = adata.obsm["spatial"]
    index = np.arange(coords.shape[0])
    genes = np.array(adata.var_names)
    GO_high_expressed = {}
    GO_high_expressed_pvals = {}
    n_go_terms = len(go_term_dict)
    for gid, (go_id, go_genes) in enumerate(go_term_dict.items()):
        if (gid + 1) % 500 == 0:
            print(f"Processed {gid + 1}/{n_go_terms}: {100. * (gid + 1)/n_go_terms}% GO terms")

        expr = adata.X[:, np.isin(genes, go_genes)].mean(axis=1)
        avg_expr = expr.mean()
        std_expr = expr.std()
        outlier_val = avg_expr + std_expr
        ind = np.array(np.where(expr > outlier_val)).flatten()
        if ind.size > 5:
            sub_coords = coords[ind, :]
            sub_dists = distance.cdist(sub_coords, sub_coords, 'euclidean')

            rand_index = np.random.choice(index, size=ind.size)
            random_coord = coords[rand_index, :]
            rand_dists = distance.cdist(random_coord, random_coord, 'euclidean')
            pval = wilcoxon(sub_dists.flatten(), rand_dists.flatten(), alternative='greater')
            if pval.pvalue < .05:
                GO_high_expressed[go_id] = ind
                GO_high_expressed_pvals[go_id] = pval.pvalue
        else:
            pass
    print(f"Found {len(GO_high_expressed)} highly expressed GO terms")
    args.spatial = True
    go_terms_w_pv = np.array([[go_id, str(GO_high_expressed_pvals[go_id])] for go_id in  sorted(GO_high_expressed_pvals.keys(), key= lambda key:GO_high_expressed_pvals[key], reverse=True)]).astype(str)
    df = pd.DataFrame(go_terms_w_pv, columns=["GO_ID", "P-Val"])

    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    high_expr_GO_out_fp = f"{output_dir}/highly_expr_go.tsv"
    df.to_csv(high_expr_GO_out_fp, sep="\t", index=False)
    print(f"Saved at {high_expr_GO_out_fp}")

    high_expr_GO_out_pkl_fp = f"{output_dir}/highly_expr_go_w_spots_indices.pkl"
    with open(high_expr_GO_out_pkl_fp, 'wb') as handle:
        pickle.dump(GO_high_expressed, handle, -1)
        print(f"Saved at {high_expr_GO_out_pkl_fp}")

def get_ovlp_GO_definitions(args, sample_name, dataset="breast_cancer"):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    high_expr_GO_out_fp = f"{output_dir}/highly_expr_go.tsv"
    df = pd.read_csv(high_expr_GO_out_fp, sep="\t", header= 0)

    fp = f'{args.dataset_dir}/Visium/Breast_Cancer/analysis/genes_with_go_ids_and_def.csv'
    go_id_def = pd.read_csv(fp).values.astype(str)
    go_dict = {go_id: go_id_def[gid, 1] for gid, go_id in enumerate(go_id_def[:, 0])}
    go_terms = df.loc[:, "GO_ID"].values.astype(str)
    go_def = np.array([go_dict[go_id] for go_id in go_terms]).astype(str)
    df["GO_DEF"] = go_def
    df = df.sort_values(by="P-Val", ascending=True)
    high_expr_GO_out_def_fp = f"{output_dir}/highly_expr_go_w_def.tsv"
    df.to_csv(high_expr_GO_out_def_fp, sep="\t", index=False)
    print(f"Saved at {high_expr_GO_out_def_fp}")

def get_clusters_annnotations(sample_name):
    if sample_name[0] == "G":
        clusters = ['APC,B,T-1', 'APC,B,T-2', 'Inva-Conn', 'Invasive-2', 'Invasive-1', 'Imm-Reg-1', 'Imm-Reg-2'
        , 'Tu.Imm.Itfc-1', 'Tu.Imm.Itfc-1', 'Tu.Imm.Itfc-1']
        return clusters
    else:
        return []

def find_ovlpd_go_terms_with_cluster_specific_go_pathways(args, sample_name, dataset="breast_cancer"):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    high_expr_GO_out_fp = f"{output_dir}/highly_expr_go.tsv"
    high_expr_go_df = pd.read_csv(high_expr_GO_out_fp, sep="\t", header=0)
    high_expr_go_terms = high_expr_go_df["GO_ID"].values.astype(str)

    cluster_dir = f'{output_dir}/cluster_specific_marker_genes'
    clusters = get_clusters_annnotations(sample_name)
    for cid, cluster in enumerate(clusters):
        cluster_go_term_fp = f"{cluster_dir}/{cluster}_topGO_terms.tsv"
        df = pd.read_csv(cluster_go_term_fp, sep="\t", header=0)
        go_ids = df["GO.ID"].values.astype(str)
        ovlp_go_ids, x_ind, y_ind = np.intersect1d(high_expr_go_terms, go_ids, return_indices=True)
        cluster_ovlp_go_terms_fp = f"{cluster_dir}/{cluster}_topGO_terms_w_high_expr_patterns.tsv"
        sub_df = df.iloc[y_ind, :]
        sub_df.to_csv(cluster_ovlp_go_terms_fp, sep="\t", index=False)
        print(f"Saved at {cluster_ovlp_go_terms_fp}")

def cell_cell_communication_preprocessing_data(args, adata):
    sc.pp.filter_genes(adata, min_counts=1)  # only consider genes with more than 1 count
    sc.pp.normalize_per_cell(adata, key_n_counts='n_counts_all', min_counts=0)  # normalize with total UMI count per cell
    sc.pp.log1p(adata)  # log transform: adata.X = log(adata.X + 1)

    genes = np.array(adata.var_names)
    cells = np.array(adata.obs_names)
    return adata, genes, cells

def save_adata_to_preprocessing_dir(args, adata_pp, sample, cells):
    pp_dir = f'{args.dataset_dir}/Visium/Breast_Cancer/preprocessed/{sample}'
    mkdir(pp_dir)

    cluster_annotations = get_clusters(args, sample)
    concat_annotations = np.transpose(np.vstack([cells, cluster_annotations]))
    annotation_fp = f'{pp_dir}/cluster_anno.tsv'
    df = pd.DataFrame(data=concat_annotations, columns=["Cell", "Annotation"])
    df.to_csv(annotation_fp, sep="\t", index=False)
    print(f"{sample} annotation saved at {annotation_fp}")

    adata_fp = f'{pp_dir}/anndata_pp.h5ad'
    mkdir(os.path.dirname(adata_fp))
    adata_pp.write(adata_fp)
    print(f"{sample} adata saved at {adata_fp}")


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

def plot_clustering_and_pseudotime(args, adata, sample_name, method="leiden", dataset="breast_cancer", scale = 1., scatter_sz=1.3, nrow = 1, annotation=False, alpha=.5):
    original_spatial = args.spatial
    args.spatial = True
    fig, axs, x, y, img, xlim, ylim = plot_hne_and_annotation(args, adata, sample_name, scale=scale, nrow=nrow, ncol=4, rsz=2.6,
                                                              csz=3.9, wspace=1, hspace=.4, annotation=annotation)
    ax = axs[1]
    ax.imshow(img, alpha=alpha)
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
    ax.imshow(img, alpha=alpha)
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
    ax.imshow(img, alpha=alpha)

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

def get_correlation_matrix_btw_clusters(args, sample_name, adata, method="cluster", dataset="breast_cancer"):
    pred_clusters = get_clusters(args, sample_name)
    uniq_clusters = np.array(np.unique(pred_clusters))
    mean_exprs = []
    for cid, uniq_cluster in enumerate(uniq_clusters):
        ind = pred_clusters == uniq_cluster
        mean_expr = adata.X[ind, :].mean(axis=0)
        mean_exprs.append(mean_expr)
    mean_exprs = np.array(mean_exprs).astype(float)
    df = pd.DataFrame(data=mean_exprs.transpose(), columns=uniq_clusters, index=np.array(adata.var_names).astype(str))
    corr_matrix = df.corr(method="pearson")

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

def plot_expr_in_ST(args, adata, genes, sample_name, fig_name, dataset="breast_cancer", scatter_sz= 6., cm = plt.get_cmap("RdPu"), n_cols = 5, max_expr_threshold=.5):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}/ST_Expr'
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
        expr = exprs[:, all_genes == gene]
        expr = (expr - expr.mean())/expr.std()
        expr_max = np.max(expr)
        expr[expr > expr_max * max_expr_threshold] = expr_max * max_expr_threshold
        # expr /= np.max(expr)
        ax = set_ax_for_expr_plotting(ax)
        # ax.imshow(img)
        st = ax.scatter(x, y, s=scatter_sz, c=expr, cmap=cm)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.invert_yaxis()
        if col == n_cols - 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            clb = fig.colorbar(st, cax=cax)
            clb.ax.set_ylabel("Z-score of Expr.", labelpad=10, rotation=270, fontsize=10, weight='bold')
        ax.set_title(gene, fontsize=12)
    fig_fp = f"{output_dir}/{fig_name}.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')


def plot_expr_in_UMAP(args, adata, genes, sample_name, fig_name, dataset="breast_cancer", scatter_sz= 6., cm = plt.get_cmap("RdPu"), n_cols = 5, max_expr_threshold=.5):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    mkdir(output_dir)
    n_genes = len(genes)
    n_rows = int(math.ceil(n_genes/n_cols))
    fig, axs = figure(n_rows, n_cols, rsz=2.2, csz=3., wspace=.2, hspace=.2)
    exprs = adata.X
    all_genes = np.array(list(adata.var_names))

    umap_fp = f"{output_dir}/umap_embeddings_by_cluster.tsv"
    df = pd.read_csv(umap_fp, sep="\t", header=0).values.astype(float)
    x, y =df[:, 0], df[:, 1]
    ylim = [-np.max(x) * 1.05, -np.min(x)]
    xlim = [-np.max(y) * 1.1, -np.min(y) * .75]
    for gid, gene in enumerate(genes):
        row = gid // n_cols
        col = gid % n_cols
        ax = axs[row][col] if n_rows > 1 else axs[col]
        expr = exprs[:, all_genes == gene]
        expr = (expr - expr.mean())/expr.std()
        expr_max = np.max(expr)
        expr[expr > expr_max * max_expr_threshold] = expr_max * max_expr_threshold
        # expr /= np.max(expr)
        ax = set_ax_for_expr_plotting(ax)
        # ax.imshow(img)
        st = ax.scatter(-y, -x, s=scatter_sz, c=expr, cmap=cm)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.invert_yaxis()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = fig.colorbar(st, cax=cax)
        clb.ax.set_ylabel("Z-score of Expr.", labelpad=10, rotation=270, fontsize=10, weight='bold')
        ax.set_title(gene, fontsize=12)
    figure_out_dir = f"{output_dir}/UMAP_Expr"
    mkdir(figure_out_dir)
    fig_fp = f"{figure_out_dir}/{fig_name}.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def set_ax_for_expr_plotting(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * .9, box.height])

    ax.invert_yaxis()
    return ax

def plot_GO_terms_w_spatial_cohr_expr(args, adata, sample_name, fig_fp, go_name, pvalue, genes_ass_go_terms, high_expr_spots_indices, dataset="breast_cancer", cm = plt.get_cmap("Reds"), scatter_sz=2):
    genes = np.array(adata.var_names)
    expr = adata.X[:, np.isin(genes, genes_ass_go_terms)].mean(axis=1)
    avg_expr = expr.mean()
    std_expr = expr.std()

    fig, ax = figure(1, 1, rsz=2.8, csz=3., wspace=.2, hspace=.2)

    fp = f'{args.dataset_dir}/Visium/Breast_Cancer/ST-imgs/{sample_name[0]}/{sample_name}/HE.jpg'
    img = plt.imread(fp)
    x, y = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    ax = set_ax_for_expr_plotting(ax)
    ax.imshow(img)
    ax.scatter(x, y, s=scatter_sz, color="#EDEDED")
    standardized_expr = (expr[high_expr_spots_indices] - avg_expr)/std_expr
    st = ax.scatter(x[high_expr_spots_indices], y[high_expr_spots_indices], s=scatter_sz, c=standardized_expr, cmap=cm)

    # standardized_expr = (expr - avg_expr) / std_expr
    # st = ax.scatter(x, y, s=scatter_sz, c=standardized_expr, cmap=cm)
    xlim = [np.min(x), np.max(x) * 1.05]
    ylim = [np.min(y) * .75, np.max(y) * 1.1]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = fig.colorbar(st, cax=cax)
    clb.ax.set_ylabel("z-score of Expr", labelpad=10, rotation=270, fontsize=10, weight='bold')
    ax.set_title(f"GO:{go_name}\np=%.2e" % (pvalue), fontsize=8)
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')
def load_high_expr_spot_indices(args, sample_name, dataset="breast_cancer"):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    high_expr_GO_out_pkl_fp = f"{output_dir}/highly_expr_go_w_spots_indices.pkl"
    with open(high_expr_GO_out_pkl_fp, 'rb') as f:
        indices = pickle.load(f)
        return indices

def plot_cluster_specific_GO_terms_w_spatial_patterns(args, adata, sample_name, go_term_dict, dataset="breast_cancer"):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    cluster_dir = f'{output_dir}/cluster_specific_marker_genes'
    clusters = get_clusters_annnotations(sample_name)
    high_expr_spots_indices = load_high_expr_spot_indices(args, sample_name)
    high_expr_GO_def = pd.read_csv(f"{output_dir}/highly_expr_go_w_def.tsv", sep="\t", header=0)
    go_pval_dict = {row["GO_ID"]: row["P-Val"] for index, row in high_expr_GO_def.iterrows()}
    for cid, cluster in enumerate(clusters):
        figure_out_dir = f"{cluster_dir}/{cluster}"
        mkdir(figure_out_dir)

        cluster_ovlp_go_terms_fp = f"{cluster_dir}/{cluster}_topGO_terms_w_high_expr_patterns.tsv"
        df = pd.read_csv(cluster_ovlp_go_terms_fp, sep="\t", header=0)
        go_ids = df["GO.ID"].values.astype(str)
        go_names = df["Term"].values.astype(str)
        for gid, go_id in enumerate(go_ids):
            print(f"Processing {cluster}: {go_id}")
            fig_out_fp = f"{figure_out_dir}/{go_id.replace(':','-')}.pdf"
            go_name = go_names[gid]
            genes_ass_go_terms = go_term_dict[go_id]
            pvalue = go_pval_dict[go_id]
            plot_GO_terms_w_spatial_cohr_expr(args, adata, sample_name, fig_out_fp, go_name, pvalue, genes_ass_go_terms, high_expr_spots_indices[go_id])

def get_clusters_and_color_dict(args, sample_name, spots_idx_dicts, original=False, dataset = "breast_cancer", method = "leiden"):
    if original:
        fp = f'{args.dataset_dir}/Visium/Breast_Cancer/ST-cluster/lbl/{sample_name}-cluster-annotation.tsv'
        df = pd.read_csv(fp, sep="\t")
        clusters = df["label"].values.astype(int)
        cluster_dict, color_dict = get_cluster_colors_and_labels_original()
        spots_in_labels = {f"{item[0]}x{item[1]}": idx for idx, item in
                           enumerate(df[["x", "y"]].values.astype(int))}
        selected_cluster_indices = np.array([val for key, val in spots_idx_dicts.items() if key in spots_in_labels]).astype(int)
        pred_clusters = np.array([cluster_dict[clusters[spots_in_labels[key]]] for key, val in spots_idx_dicts.items() if key in spots_in_labels])
        color_dict = {cluster_dict[cluster]: cluster_color for cluster, cluster_color in color_dict.items()}
    else:
        args.spatial = True
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
        pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(str)
        uniq_pred = np.unique(pred_clusters)
        cluster_color_dict = get_cluster_colors(args, sample_name)
        unique_cluster_dict = {cluster: cluster_color_dict[cluster]["abbr"] for cluster in cluster_color_dict.keys()}
        color_dict = {}
        for cid, cluster in enumerate(uniq_pred):
            label = unique_cluster_dict[int(cluster)]
            color_dict[label] = f"#{cluster_color_dict[int(cluster)]['color']}"

        pred_clusters = get_clusters(args, sample_name)
        selected_cluster_indices = np.arange(len(pred_clusters))
    return pred_clusters, color_dict, selected_cluster_indices

def plot_umap_tsne_phate(args, sample_name, adata, pred_clusters, color_dict, original=False, n_neighbors=6, ncol = 3, scatter_sz= 4, method="leiden", dataset="breast_cancer", embedding=True, save_umap=False):

    fig, axs = figure(1, ncol, rsz=2.8, csz=4.4, wspace=.4, hspace=.2)

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.leiden(adata, resolution=.5)

    sc.tl.umap(adata)
    sc.tl.tsne(adata, perplexity=20, use_rep='X')

    phate_op = phate.PHATE(k=n_neighbors, t=80, gamma=1)
    data_phate = phate_op.fit_transform(adata.X)
    positions = [adata.obsm["X_umap"], adata.obsm["X_tsne"], data_phate]
    vis_names = ["UMAP", "t-SNE", "PHATE"]
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'

    for idx in range(ncol):
        ax = axs[idx]
        name = vis_names[idx]
        ax.axis('off')
        uniq_clusters = np.array(np.unique(pred_clusters))

        for cid, uniq_cluster in enumerate(uniq_clusters):
            position = positions[idx][pred_clusters == uniq_cluster]
            ax.scatter(position[:, 0], position[:, 1], s=scatter_sz, color=color_dict[uniq_cluster], label=uniq_cluster)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5))
        for handle in lgnd.legendHandles:
            handle._sizes = [10]
    name = "embedding" if embedding else "expr"
    suff = "origin_label" if original else "cluster"
    fig_fp = f"{output_dir}/{name}_visualization_lbl_by_{suff}.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

    if save_umap:
        umap_fp = f"{output_dir}/umap_embeddings_by_{suff}.tsv"
        df = pd.DataFrame(data=np.array(adata.obsm["X_umap"]).astype(float), columns=["UMAP_1", "UMAP_2"])
        df.to_csv(umap_fp, sep="\t", index=False)

def plot_umap_tsne_phate_color_by_pseudotime(args, sample_name, adata, n_neighbors=6, ncol = 3, scatter_sz= 4, method="leiden", dataset="breast_cancer", embedding=True):

    fig, axs = figure(1, ncol, rsz=2.8, csz=4.4, wspace=.4, hspace=.2)

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.leiden(adata, resolution=.5)

    sc.tl.umap(adata)
    sc.tl.tsne(adata, perplexity=20, use_rep='X')

    phate_op = phate.PHATE(k=n_neighbors, t=80, gamma=1)
    data_phate = phate_op.fit_transform(adata.X)
    positions = [adata.obsm["X_umap"], adata.obsm["X_tsne"], data_phate]
    vis_names = ["UMAP", "t-SNE", "PHATE"]
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'

    pseudotimes = pd.read_csv(f"{output_dir}/pseudotime.tsv", header=None).values.flatten().astype(float)
    pseudo_time_cm = plt.get_cmap("gist_rainbow")

    for idx in range(ncol):
        ax = axs[idx]
        name = vis_names[idx]
        ax.axis('off')
        st = ax.scatter(positions[idx][:, 0], positions[idx][:, 1], s=scatter_sz, c=pseudotimes, cmap=pseudo_time_cm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%")
        clb = fig.colorbar(st, cax=cax)
        clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=8, weight='bold')
        ax.set_title(name, fontsize=title_sz)
    name = "embedding" if embedding else "expr"
    fig_fp = f"{output_dir}/{name}_latent_space_color_by_pseudotime.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')


####################################
#-------------Pipelines------------#
####################################

def train_pipeline(args, adata, sample_name, dataset="breast_cancer", clustering_method="leiden", resolution=.5, n_neighbors=6, isTrain=True):
    adata_filtered, spatial_graph = preprocessing_data(args, adata)
    if isTrain:
        for spatial in [False, True]:
            args.spatial = spatial
            embedding = train(args, adata_filtered, spatial_graph)
            save_features(args, embedding, dataset, sample_name)
            clustering(args, dataset, sample_name, clustering_method, n_neighbors=n_neighbors, resolution=resolution)
            pseudotime(args, dataset, sample_name, root_cell_type=None, cell_types=None, n_neighbors=n_neighbors, resolution=resolution)
    return adata_filtered

def go_pipeline(args):
    letters = ["G"]  # ["A", "B", "C", "D", "E", "F", "G", "H"]#["H"]#
    n_samples = [1 for letter in letters]  # [6, 6, 6, 6, 3, 3, 3, 3]#
    sample_list = [f"{letter}{sid}" for lid, letter in enumerate(letters) for sid in range(1, n_samples[lid] + 1)]
    go_term_dict = get_GO_term_dict(args)
    for sample_name in sample_list:
        adata, _ = load_breast_cancer_data(args, sample_name)
        # get_GO_terms_with_spatial_coherent_expr(args, adata, sample_name, go_term_dict)
        # get_ovlp_GO_definitions(args, sample_name)
        # find_ovlpd_go_terms_with_cluster_specific_go_pathways(args, sample_name)
        # plot_cluster_specific_GO_terms_w_spatial_patterns(args, adata, sample_name, go_term_dict)

def umap_pipeline(args):
    letters = ["G"]  # ["A", "B", "C", "D", "E", "F", "G", "H"]#["H"]#
    n_samples = [1 for letter in letters]  # [6, 6, 6, 6, 3, 3, 3, 3]#
    sample_list = [f"{letter}{sid}" for lid, letter in enumerate(letters) for sid in range(1, n_samples[lid] + 1)]

    for sample_name in sample_list:
        adata, spots_idx_dicts = load_breast_cancer_data(args, sample_name)
        adata_filtered = train_pipeline(args, adata, sample_name, n_neighbors=5, isTrain=False)

        original = False
        adata_embed = get_adata_from_embeddings(args, sample_name)
        adata_expr = adata_filtered
        pred_clusters, color_dict_for_cluster, selected_cluster_indices = get_clusters_and_color_dict(args, sample_name,
                                                                                                      spots_idx_dicts,
                                                                                                      original)
        plot_umap_tsne_phate(args, sample_name, adata_embed, pred_clusters, color_dict_for_cluster, original=original,
                             embedding=True, save_umap=False)
        # plot_umap_tsne_phate_color_by_pseudotime(args, sample_name, adata_embed, embedding=True)
        # plot_umap_tsne_phate(args, sample_name, adata_expr, pred_clusters, color_dict_for_cluster, original=original,
        #                      embedding=False)
        #
        # original = True
        # pred_clusters, color_dict_for_cluster, selected_cluster_indices = get_clusters_and_color_dict(args, sample_name, spots_idx_dicts, original)
        #
        # adata_embed = adata_embed[selected_cluster_indices, :]
        # adata_expr = adata[selected_cluster_indices, :]
        #
        # plot_umap_tsne_phate(args, sample_name, adata_embed, pred_clusters, color_dict_for_cluster, original=original, embedding=True)
        # plot_umap_tsne_phate(args, sample_name, adata_expr, pred_clusters, color_dict_for_cluster, original=original, embedding=False)

def basic_pipeline(args):
    letters = ["G"]#["A", "B", "C", "D", "E", "F", "G", "H"]#["H"]#
    n_samples = [1 for letter in letters] #[6, 6, 6, 6, 3, 3, 3, 3]#
    sample_list = [f"{letter}{sid}" for lid, letter in enumerate(letters) for sid in range(1, n_samples[lid] + 1)]

    for sample_name in sample_list:
        adata, spots_idx_dicts = load_breast_cancer_data(args, sample_name)
        adata_filtered = train_pipeline(args, adata, sample_name, n_neighbors=5, isTrain=False)
        # plot_clustering(args, adata, sample_name, scatter_sz=3, annotation=False, scale=1)
        # plot_pseudotime(args, adata, sample_name)
        plot_rank_marker_genes_group(args, sample_name, adata_filtered, top_n_genes=5)
        # get_correlation_matrix_btw_clusters(args, sample_name, adata_filtered)

def figure_pipeline(args):
    sample_list = ["G1"]
    for sample_name in sample_list:
        adata, _ = load_breast_cancer_data(args, sample_name)
        plot_clustering_and_pseudotime(args, adata, sample_name, scatter_sz=5)
        # adata_filtered = train_pipeline(args, adata, sample_name, n_neighbors=5, isTrain=False)
        # plot_clustering(args, adata, sample_name, scatter_sz=3, annotation=False, scale=1)
        # plot_pseudotime(args, adata, sample_name)

def calc_pseudotime_corr_genes(args, adata, sample_name, dataset, n_top=16):
    original_spatial = args.spatial
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    pseudotimes = pd.read_csv(f"{output_dir}/pseudotime.tsv", header=None).values.flatten().astype(float)
    adata, _ = preprocessing_data(args, adata)
    expr = adata.X
    genes = np.array(adata.var_names)
    gene_corrs = [[gene] + list(pearsonr(expr[:, gix].flatten(), pseudotimes)) for gix, gene in enumerate(genes)]
    gene_corrs.sort(key=lambda k:k[-1])
    df = pd.DataFrame(gene_corrs, columns=["gene", "corr", "p-val"])
    df.to_csv(f"{output_dir}/Gene_Corr_with_PST.tsv", index=False)
    args.spatial = original_spatial
    return df.values[:n_top, 0].astype(str)

def corr_expr_analysis_pipeline(args):
    sample_list = ["G1"]
    for sample_name in sample_list:
        adata, _ = load_breast_cancer_data(args, sample_name)
        genes_corred = calc_pseudotime_corr_genes(args, adata, sample_name, "breast_cancer",n_top=20)
        plot_expr_in_ST(args, adata, genes_corred, sample_name, "Genes_Corr_wt_PST", scatter_sz=7)

def expr_analysis_pipeline(args):
    sample_list = ["G1"]
    for sample_name in sample_list:
        adata, _ = load_breast_cancer_data(args, sample_name)
        # save_cluster_specific_genes(args, adata, sample_name, "cluster")
        cluster_specific_genes_dict = get_top_n_cluster_specific_genes(args, sample_name, "cluster", top_n=5)
        for cs_name, cs_genes in cluster_specific_genes_dict.items():
            if cs_name == "In situ cancer-1":
                cs_genes[-1] = 'ERBB2'
                cs_genes[-2] = 'VIM'
            elif cs_name == "Invasive-1":
                cs_genes[-1] = 'FAP'
                cs_genes[-2] = 'S100A4'
                cs_genes[-3] = 'MCAM'
                cs_genes[-4] = 'VEGFA'
                cs_genes[0] = 'BAMBI'

            plot_expr_in_ST(args, adata, cs_genes, sample_name, cs_name, scatter_sz=7)
            # plot_expr_in_UMAP(args, adata, cs_genes, sample_name, cs_name, scatter_sz=7)

def cell_cell_communication_prep_pipeline(args):
    sample_list = ["G1"]
    for sid, sample in enumerate(sample_list):
        adata, _ = load_breast_cancer_data(args, sample)
        adata_pp, genes, cells = cell_cell_communication_preprocessing_data(args, adata)
        save_adata_to_preprocessing_dir(args, adata_pp, sample, cells)