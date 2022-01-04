# -*- coding:utf-8 -*-
import math
import anndata
from scipy.spatial import distance_matrix
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

def get_annotations_dlpfc(args, sample_name, dataset="DLPFC"):
    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}'
    df_meta = pd.read_csv(f"{data_root}/metadata.tsv", sep='\t')
    anno_clusters = df_meta['layer_guess'].values.astype(str)
    return anno_clusters

def write_list_to_file(fp, arr):
    file = open(fp, "w")
    file.write("\n".join([str(ele) for ele in arr]) + "\n")
    file.close()
    print(f"Write arr to {fp} successful!")


def plot_hne_and_annotation(args, sample_name, dataset="DLPFC", nrow = 1, HnE=False, cm = plt.get_cmap("plasma_r"), scale = 0.045, scatter_sz=1.3, ncol=4, rsz=2.5, csz=2.8, wspace=.4, hspace=.5):
    subfig_offset = 1 if HnE else 0

    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}'
    adata = load_ST_file(data_root)
    coord = adata.obsm['spatial'].astype(float) * scale
    x, y = coord[:, 0], coord[:, 1]
    anno_clusters = get_annotations_dlpfc(args, sample_name)
    img = plt.imread(f"{data_root}/spatial/tissue_lowres_image.png")
    xlimits = [130, 550]
    ylimits = [100, 530]
    fig, axs = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace)
    if nrow == 1:
        for ax in axs:
            ax.axis('off')
            ax.set_xlim(xlimits)
            ax.set_ylim(ylimits)
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

def plot_clustering(args, sample_name, method="leiden", dataset="DLPFC", HnE=False, cm = plt.get_cmap("plasma"), scale = 0.045, scatter_sz=1.3, nrow = 1):
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

def plot_clustering_comparison(args, sample_name, method="leiden", dataset="DLPFC", HnE=False, cm = plt.get_cmap("plasma"), scale = 0.045, scatter_sz=1.3, nrow = 2, ncol=4):
    methods = ["Seurat", "Giotto", "stLearn", "SpaGCN", "BayesSpace"]
    col_names = ["seurat_clusters", "HMRF_cluster", "X_pca_kmeans", "refined_pred", "spatial.cluster"]
    original_spatial = args.spatial
    fig, axs, x, y, subfig_offset = plot_hne_and_annotation(args, sample_name, HnE=HnE, cm=cm, scale=scale, scatter_sz=scatter_sz, nrow=nrow, ncol=ncol, rsz=2.9, csz=2.9, wspace=.3, hspace=.3)
    for mid, benchmarking_method in enumerate(methods):
        print(f"Processing {benchmarking_method}")
        real_ind = mid + subfig_offset + 1
        row = real_ind // ncol
        col = real_ind % ncol
        ax = axs[row][col] if nrow > 1 else axs[col]
        input_fp = f'{args.output_dir}/{dataset}/{sample_name}/{benchmarking_method}/metadata.tsv'
        df = pd.read_csv(input_fp, sep='\t')
        pred_clusters = df[col_names[mid]].values.astype(int)
        uniq_pred = np.unique(pred_clusters)
        n_cluster = len(uniq_pred)
        for cid, cluster in enumerate(uniq_pred):
            color = cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
            ind = pred_clusters == cluster
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster)
        ax.set_title(benchmarking_method, fontsize=title_sz, pad=-20)
    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        title = args.arch if not spatial else "%s + SP" % args.arch
        print(f"Processing {title}")
        real_ind = sid + subfig_offset + len(methods) + 1
        row = real_ind // ncol
        col = real_ind % ncol
        ax = axs[row][col] if nrow > 1 else axs[col]
        args.spatial = spatial
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
        pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(int)
        uniq_pred = np.unique(pred_clusters)
        n_cluster = len(uniq_pred)
        for cid, cluster in enumerate(uniq_pred):
            color = cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
            ind = pred_clusters == cluster
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster)
        ax.set_title(title, fontsize=title_sz, pad=-20)
        if sid:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            lgnd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, scatterpoints=1, handletextpad=0.1, borderaxespad=.1)
            for handle in lgnd.legendHandles:
                handle._sizes = [8]
    fig_fp = f"{output_dir}/{method}_comparison.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')
    args.spatial = original_spatial

def plot_pseudotime(args, sample_name, dataset="DLPFC", HnE=False, cm = plt.get_cmap("gist_rainbow"), scale = 0.045):
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

def plot_pseudotime_comparison(args, sample_name, dataset="DLPFC", cm = plt.get_cmap("gist_rainbow"), scale = 0.045, n_neighbors=50, root_cell_type = None, cell_types=None):
    methods = ["Seurat", "stLearn", "DGI", "DGI_SP"]
    files = ["seurat.PCs.tsv", "PCs.tsv", "features.tsv", "features.tsv"]
    nrow, ncol = 1, len(methods)

    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}'
    adata = load_ST_file(data_root)
    coord = adata.obsm['spatial'].astype(float) * scale
    x, y = coord[:, 0], coord[:, 1]

    xlimits = [130, 550]
    ylimits = [100, 530]

    fig, axs = figure(nrow, ncol, rsz=2.4, csz=2.8, wspace=.3, hspace=.3)
    for ax in axs:
        ax.axis('off')
        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)
        ax.invert_yaxis()
    for mid, method in enumerate(methods):
        print(f"Processing {sample_name} {method}")
        col = mid % ncol
        ax = axs[col]
        output_dir = f'{args.output_dir}/{dataset}/{sample_name}/{method}'
        pseudotime_fp = f"{output_dir}/pseudotime.tsv"
        if not os.path.exists(pseudotime_fp):
            file_name = files[mid]
            feature_fp = f'{output_dir}/{file_name}'
            if file_name.endswith("npz"):
                obj = np.load(feature_fp)
                adata = anndata.AnnData(obj.f.sedr_feat)
            else:
                adata = sc.read_csv(feature_fp, delimiter="\t")
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
            sc.tl.umap(adata)
            sc.tl.leiden(adata, resolution=.8)
            sc.tl.paga(adata)
            distances = distance_matrix(adata.X, adata.X)
            sum_dists = distances.sum(axis=1)
            adata.uns['iroot'] = np.argmax(sum_dists)
            if root_cell_type:
                descend_dist_inds = sorted(range(len(sum_dists)), key=lambda k: sum_dists[k], reverse=True)
                for root_idx in descend_dist_inds:
                    if cell_types[root_idx] == root_cell_type:
                        adata.uns['iroot'] = root_idx
                        break
            sc.tl.diffmap(adata)
            sc.tl.dpt(adata)
            pseudotimes = adata.obs['dpt_pseudotime'].to_numpy()
            np.savetxt(pseudotime_fp, pseudotimes, fmt='%.5f', header='', footer='', comments='')
            print("Saved %s succesful!" % pseudotime_fp)
        else:
            pseudotimes = pd.read_csv(pseudotime_fp, header=None).values.flatten().astype(float)
        st = ax.scatter(x, y, s=1, c=pseudotimes, cmap=cm)
        if col == (ncol - 1):
            axins = inset_locator.inset_axes(ax, width="5%", height="60%",  loc='lower left', bbox_to_anchor=(1.05, 0.1, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
            clb = fig.colorbar(st, cax=axins)
            clb.set_ticks([0.0, 1.0])
            clb.set_ticklabels(["0", "1"])
            clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
        ax.set_title(method.replace("_", " + "), fontsize=title_sz)
    fig_fp = f"{output_dir}/psudotime_comparison.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_umap_comparison(args, sample_name, dataset="DLPFC", n_neighbors=50):
    methods = ["Seurat",  "stLearn", "DGI", "DGI_SP"]
    files = ["seurat.PCs.tsv", "PCs.tsv", "features.tsv", "features.tsv"]
    nrow, ncol = 1, len(methods)

    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}'
    fig, axs = figure(nrow, ncol, rsz=2.4, csz=2.8, wspace=.3, hspace=.3)
    for ax in axs:
        ax.axis('off')

    for mid, method in enumerate(methods):
        print(f"Processing {sample_name} {method}")
        col = mid % ncol
        ax = axs[col]
        output_dir = f'{args.output_dir}/{dataset}/{sample_name}/{method}'
        umap_positions_fp = f"{output_dir}/umap_positions.tsv"
        if not os.path.exists(umap_positions_fp):
            file_name = files[mid]
            feature_fp = f'{output_dir}/{file_name}'
            if file_name.endswith("npz"):
                obj = np.load(feature_fp)
                adata = anndata.AnnData(obj.f.sedr_feat)
            else:
                adata = sc.read_csv(feature_fp, delimiter="\t")
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
            sc.tl.umap(adata)
            umap_positions = adata.obsm["X_umap"]
            np.savetxt(umap_positions_fp, umap_positions, fmt='%.5f\t%.5f', header='', footer='', comments='')
        else:
            umap_positions = pd.read_csv(umap_positions_fp, header=None, sep="\t").values.astype(float)
        df_meta = pd.read_csv(f'{data_root}/metadata.tsv', sep='\t')
        annotations = df_meta['layer_guess'].values.astype(str)
        cluster_names = list(np.unique(annotations))

        cm = plt.get_cmap("tab10")
        for cid, cluster in enumerate(cluster_names[:-1]):
            umap_sub = umap_positions[annotations == cluster]
            color = cm(1. * cid / (len(cluster_names) + 1))
            ax.scatter(umap_sub[:, 0], umap_sub[:, 1], s=2, color=color, label=cluster)
        if mid == len(methods) - 1:
            box = ax.get_position()
            height_ratio = 1.0
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * height_ratio])
            ax.legend(loc='center left', fontsize='x-small', bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.05,
                      borderaxespad=.1)
        ax.set_title(method.replace("_", " + "), fontsize=title_sz)
    fig_fp = f"{output_dir}/umap_comparison.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')


def rank_marker_genes_group(args, sample_name, method="leiden", dataset="DLPFC"):
    original_spatial = args.spatial
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    adata = load_DLPFC_data(args, sample_name, v2=False)
    adata_filtered, _, _, _, _, _ = preprocessing_data(args, adata)
    pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(str)
    adata_filtered.obs["leiden"] = pd.Categorical(pred_clusters)
    sc.tl.rank_genes_groups(adata_filtered, 'leiden', method='wilcoxon')
    sc.pl.rank_genes_groups(adata_filtered, n_genes=25, ncols=5, fontsize=10, sharey=False, save=True)
    sc.pl.rank_genes_groups_heatmap(adata_filtered, n_genes=2, standard_scale='var', save="heatmap.pdf")
    sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=2, standard_scale='var', save="mean_expr.pdf")
    sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=2, values_to_plot="logfoldchanges", cmap='bwr', vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change', save="dot_lfc.pdf")
    sc.pl.rank_genes_groups_matrixplot(adata_filtered, n_genes=2, values_to_plot="logfoldchanges", cmap='bwr', vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change', save="matrix_lfc.pdf")
    args.spatial = original_spatial
    cluster_marker_genes_fp = f'{output_dir}/marker_genes_pval.tsv'
    result = adata_filtered.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    df = pd.DataFrame(
        {group + '_' + key[:1]: result[key][group]
        for group in groups for key in ['names', 'pvals']})
    df.to_csv(cluster_marker_genes_fp, sep="\t", index=False)

    gene_back_ground = list(adata_filtered.var_names)
    gene_back_ground_fp = f"{output_dir}/gene_back_ground.txt"
    write_list_to_file(gene_back_ground_fp, gene_back_ground)

def plot_marker_gene_expression(args, sample_name, gene_names, dataset="DLPFC", ncol = 4, scale = 0.045, cm = plt.get_cmap("plasma")):
    original_spatial = args.spatial
    args.spatial = True
    nrow = int(math.ceil(len(gene_names)/float(ncol)))

    adata = load_DLPFC_data(args, sample_name, v2=False)
    coord = adata.obsm['spatial'].astype(float) * scale
    x, y = coord[:, 0], coord[:, 1]
    adata_filtered, _, genes, _, _, _ = preprocessing_data(args, adata)
    expr = np.asarray(adata.X.todense())
    fig, axs = figure(nrow, ncol, rsz=2.5, csz=2.6, wspace=.2, hspace=.2)
    xlimits = [130, 550]
    ylimits = [100, 530]
    for gid, gene in enumerate(gene_names):
        row = gid // ncol
        col = gid % ncol
        ax = axs[row][col] if nrow > 1 else axs[col]
        ax.axis('off')
        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)
        ax.invert_yaxis()
        ax.set_title(gene)
        gene_expr = expr[:, genes.index(gene)].flatten()
        gene_expr /= np.max(gene_expr)
        st = ax.scatter(x, y, s=1.3, c=gene_expr, cmap=cm)
        ax.set_title(gene, fontsize=14, pad=-30)
        if col == (ncol - 1):
            axins = inset_locator.inset_axes(ax, width="7%", height="50%",  loc='lower left', bbox_to_anchor=(1.05, 0.1, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
            clb = fig.colorbar(st, cax=axins)
            clb.set_ticks([0.0, 1.0])
            clb.set_ticklabels(["Min", "Max"])

    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    fig_fp = f"{output_dir}/marker_gene_expr.pdf"
    fig.savefig(fig_fp, dpi=300)
    plt.close('all')
    args.spatial = original_spatial

