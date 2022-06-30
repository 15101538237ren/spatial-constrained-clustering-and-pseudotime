# -*- coding:utf-8 -*-
import math, shutil
import cmcrameri as cmc
from colour import Color
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr
from python_codes.util.util import *
from matplotlib.colors import to_hex
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial','Roboto']
rcParams['savefig.dpi'] = 300

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
from python_codes.util.exchangeable_loom import write_exchangeable_loom
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

def figure(nrow, ncol, rsz=3., csz=3., wspace=.4, hspace=.5, bottom=None):
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * csz, nrow * rsz))
    plt_setting()
    plt.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom)
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
    lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(.93, 0.5), scatterpoints=1, handletextpad=0.1, borderaxespad=.1)
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

def plot_clustering_comparison(args, sample_name, method="leiden", dataset="DLPFC", HnE=False, cm = plt.get_cmap("plasma"), scale = 0.045, scatter_sz=1.3, nrow = 2, ncol=4): #plt.get_cmap("rainbow")
    methods = ["Seurat","Giotto", "stLearn", "BayesSpace", "MERINGUE"] #, "SpaGCN"
    col_names = ["seurat_clusters","HMRF_cluster", "X_pca_kmeans", "spatial.cluster", "clusters"] #, "refined_pred"
    original_spatial = args.spatial
    fig, axs, x, y, subfig_offset = plot_hne_and_annotation(args, sample_name, HnE=HnE, cm=cm, scale=scale, scatter_sz=scatter_sz, nrow=nrow, ncol=ncol, rsz=2.7, csz=2.8, wspace=.3, hspace=.3)
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
        title = "SpaceFlow" if spatial else title
        ax.set_title(title, fontsize=title_sz, pad=-20)
        if sid:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            lgnd = ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=8, scatterpoints=1, handletextpad=0.1, borderaxespad=.1)
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

def calc_pseudotime_corr_genes(args, sample_name, dataset="DLPFC", n_top=28):
    original_spatial = args.spatial
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    pseudotimes = pd.read_csv(f"{output_dir}/pseudotime.tsv", header=None).values.flatten().astype(float)
    adata = load_DLPFC_data(args, sample_name, v2=False)
    adata, _ = preprocessing_data(args, adata)
    expr = np.asarray(adata.X.todense())
    genes = np.array(adata.var_names)
    gene_corrs = [[gene] + list(pearsonr(expr[:, gix].flatten(), pseudotimes)) for gix, gene in enumerate(genes)]
    gene_corrs.sort(key=lambda k:k[-1])
    df = pd.DataFrame(gene_corrs, columns=["gene", "corr", "p-val"])
    df.to_csv(f"{output_dir}/Gene_Corr_with_PST.tsv", index=False)
    args.spatial = original_spatial
    return df.values[:n_top, 0].astype(str)

def plot_pseudotime_comparison(args, sample_name, dataset="DLPFC", cm = cmc.cm.roma, scale = 0.045, n_neighbors=50, root_cell_type = None, cell_types=None):##plt.get_cmap("gist_rainbow")
    methods = ["Seurat", "monocle", "stLearn", "DGI_SP"]#, "slingshot", "DGI"
    files = ["seurat.PCs.tsv", None, "PCs.tsv", "features.tsv"]#, None, "features.tsv"
    nrow, ncol = 1, len(methods)

    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}'
    adata = load_ST_file(data_root)
    coord = adata.obsm['spatial'].astype(float) * scale
    x, y = coord[:, 0], coord[:, 1]

    xlimits = [130, 550]
    ylimits = [100, 530]

    fig, axs = figure(nrow, ncol, rsz=2.4, csz=2.8, wspace=.35, hspace=.3)
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
        axins = inset_locator.inset_axes(ax, width="5%", height="60%",  loc='lower left', bbox_to_anchor=(1.05, 0.1, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        clb = fig.colorbar(st, cax=axins)
        clb.set_ticks([0.0, np.max(pseudotimes)])
        clb.set_ticklabels(["0", "1"])
        label = "pSM value" if col == ncol - 1 else "pseudotime"
        clb.ax.set_ylabel(label, labelpad=5, rotation=270, fontsize=10, weight='bold')
        # method = "SpaceFlow" if mid == len(methods) - 1 else method
        # method = method.capitalize() if method != "stLearn" else method
        # ax.set_title(method.replace("_", " + "), fontsize=title_sz, pad=-10)
    fig_fp = f"{output_dir}/psudotime_comparison.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_umap_comparison(args, sample_name, dataset="DLPFC", n_neighbors=50):
    methods = ["Seurat",  "stLearn", "DGI", "DGI_SP"]
    files = ["seurat.PCs.tsv", "PCs.tsv", "features.tsv", "features.tsv"]
    nrow, ncol = 1, len(methods)

    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}'
    fig, axs = figure(nrow, ncol, rsz=3.2, csz=3.6, wspace=.3, hspace=.3)
    for ax in axs:
        ax.axis('off')
    cm = plt.get_cmap("tab20")

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
        #n_cluster = len(cluster_names[:-1])
        for cid, cluster in enumerate(cluster_names[:-1]):
            umap_sub = umap_positions[annotations == cluster]
            color = cm(1. * cid / (len(cluster_names) + 1))
            #color = cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
            ax.scatter(umap_sub[:, 0], umap_sub[:, 1], s=1, color=color, label=cluster)
        if mid == len(methods) - 1:
            box = ax.get_position()
            height_ratio = 1.0
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height * height_ratio])
            lgnd = ax.legend(loc='center left', fontsize=10, bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.05,
                      borderaxespad=.1)
            for handle in lgnd.legendHandles:
                handle._sizes = [12]
        method = "SpaceFlow" if method == "DGI_SP" else method
        ax.set_title(method.replace("_", " + "), fontsize=title_sz)
    fig_fp = f"{output_dir}/umap_comparison.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_umap_comparison_with_coord_alpha(args, sample_name, dataset="DLPFC", n_neighbors=50):
    methods = ["Seurat",  "stLearn", "DGI", "DGI_SP"]
    files = ["seurat.PCs.tsv", "PCs.tsv", "features.tsv", "features.tsv"]
    nrow, ncol = 1, len(methods)
    adata_origin = load_DLPFC_data(args, sample_name, v2=False)
    coord = adata_origin.obsm['spatial'].astype(float)
    x, y = coord[:, 0], coord[:, 1]
    normed_x = (x - np.min(x))/(np.max(x) - np.min(x))
    normed_y = (y - np.min(y))/(np.max(y) - np.min(y))
    normed_c = np.sqrt(normed_x**2 + normed_y**2)
    normed_c = (normed_c - np.min(normed_c))/(np.max(normed_c) - np.min(normed_c))
    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}'
    fig, axs = figure(nrow, ncol, rsz=3, csz=3.6, wspace=.3, hspace=.3)
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
            ind = annotations == cluster
            umap_sub = umap_positions[ind]
            alphas = normed_c[ind]
            color = to_hex(cm(1. * cid / (len(cluster_names) + 1)))
            color_gradients = linear_gradient(color, n=6)["hex"]
            n = umap_sub.shape[0]
            colors = np.array([color_gradients[int(alphas[i] // 0.2) + 1] for i in range(n)])
            ax.scatter(umap_sub[:, 0], umap_sub[:, 1], s=1, color=colors, label=cluster)

        method = "SpaceFlow" if method == "DGI_SP" else method
        ax.set_title(method.replace("_", " + "), fontsize=title_sz, pad=10)
    fig_fp = f"{output_dir}/umap_comparison-calpha.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_umap_comparison_with_coord_gradient(args, sample_name, dataset="DLPFC", n_neighbors=50):
    methods = ["Seurat","stLearn", "DGI", "DGI_SP"]#
    files = ["seurat.PCs.tsv", "PCs.tsv", "features.tsv", "features.tsv"]#
    nrow, ncol = 1, len(methods) + 1
    adata_origin = load_DLPFC_data(args, sample_name, v2=False)
    coord = adata_origin.obsm['spatial'].astype(float)
    x, y = coord[:, 0], coord[:, 1]
    # ncells = x.shape[0]
    normed_x = (x - np.min(x))/(np.max(x) - np.min(x))
    normed_y = (y - np.min(y))/(np.max(y) - np.min(y))
    normed_c = np.sqrt(normed_x**2 + normed_y**2)
    dist_to_central = (normed_x - normed_y)
    normed_dist_to_central = (dist_to_central - np.min(dist_to_central))/(np.max(dist_to_central) - np.min(dist_to_central))
    normed_c = (normed_c - np.min(normed_c))/(np.max(normed_c) - np.min(normed_c))
    normed_c = normed_c#normed_dist_to_central
    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}'
    fig, axs = figure(nrow, ncol, rsz=3, csz=3.7, wspace=.4, hspace=.1, bottom=.1)
    for ax in axs:
        ax.axis('off')
    cm = cmc.cm.bam#plt.get_cmap("gist_ncar")
    ax = axs[0]
    st = ax.scatter(x, y, s=2, c=normed_c, cmap=cm)
    #ax.set_title("Cell Locations", fontsize=title_sz + 4, pad=10)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    axins = inset_locator.inset_axes(ax, width="5%", height="80%", loc='lower left',
                                     bbox_to_anchor=(1.05, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    clb = fig.colorbar(st, cax=axins)
    clb.set_ticks([np.nanmin(normed_c), np.nanmax(normed_c)])
    clb.set_ticklabels(["Min", "Max"])
    clb.ax.tick_params(labelsize=14)
    clb.ax.set_ylabel("Dist. to Origin", labelpad=-5, rotation=270, fontsize=14, weight='bold')

    for mid, method in enumerate(methods):
        print(f"Processing {sample_name} {method}")
        col = mid % ncol
        ax = axs[col + 1]
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

        for cid, cluster in enumerate(cluster_names[:-1]):
            ind = annotations == cluster
            umap_sub = umap_positions[ind]
            st = ax.scatter(umap_sub[:, 0], umap_sub[:, 1], s=1, c=normed_c[ind], label=cluster, cmap=cm)
        method = "SpaceFlow" if method == "DGI_SP" else method
        #ax.set_title(f"{method}", fontsize=title_sz+ 4, pad=10)
    fig_fp = f"{output_dir}/umap_comparison-coord_gradient.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def rank_marker_genes_group(args, sample_name, method="leiden", dataset="DLPFC", top_n_genes=3):
    original_spatial = args.spatial
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    adata = load_DLPFC_data(args, sample_name, v2=False)
    adata_filtered, _ = preprocessing_data(args, adata)
    pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(str)
    adata_filtered.obs["leiden"] = pd.Categorical(pred_clusters)
    sc.tl.rank_genes_groups(adata_filtered, method, method='wilcoxon')
    sc.pl.rank_genes_groups(adata_filtered, n_genes=25, ncols=5, fontsize=10, sharey=False,
                            save=f"{sample_name}_ranks_gby_{method}.pdf")
    sc.pl.rank_genes_groups_heatmap(adata_filtered, n_genes=top_n_genes, standard_scale='var', show_gene_labels=True,
                                    save=f"{sample_name}_heatmap_gby_{method}.pdf")
    sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=top_n_genes, standard_scale='var', cmap='bwr',
                                    save=f"{sample_name}_mean_expr_gby_{method}.pdf")
    sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=top_n_genes, values_to_plot="logfoldchanges", cmap='bwr',
                                    vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change',
                                    save=f"{sample_name}_dot_lfc_gby_{method}.pdf")
    sc.pl.rank_genes_groups_matrixplot(adata_filtered, n_genes=top_n_genes, values_to_plot="logfoldchanges", cmap='bwr',
                                       vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change',
                                       save=f"{sample_name}_matrix_lfc_gby_{method}.pdf")
    sc.pl.rank_genes_groups_matrixplot(adata_filtered, n_genes=top_n_genes, cmap='bwr', colorbar_title='Mean Expr.',
                                       save=f"{sample_name}_matrix_mean_expr_gby_{method}.pdf")
    args.spatial = original_spatial

    files = [f"rank_genes_groups_{method}{sample_name}_ranks_gby_{method}.pdf",
             f"heatmap{sample_name}_heatmap_gby_{method}.pdf",
             f"dotplot_{sample_name}_mean_expr_gby_{method}.pdf",
             f"dotplot_{sample_name}_dot_lfc_gby_{method}.pdf",
             f"matrixplot_{sample_name}_matrix_lfc_gby_{method}.pdf",
             f"matrixplot_{sample_name}_matrix_mean_expr_gby_{method}.pdf"]

    for file in files:
        src_fp = f"./figures/{file}"
        target_fp = f"{output_dir}/{file}"
        shutil.move(src_fp, target_fp)

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

def export_pca_and_cluster_pipeline_for_slingshot(args, dataset = "DLPFC", sample_name="151671"):
    dataset_dir = f"{args.dataset_dir}/{dataset}/{sample_name}"
    adata = load_DLPFC_data(args, sample_name, v2=False)
    adata_filtered, _, = preprocessing_data(args, adata)

    output_dir = f'{dataset_dir}/export'
    mkdir(output_dir)

    #write_exchangeable_loom(adata_filtered, f'{output_dir}/adata_filtered.loom')

    locs = pd.DataFrame(adata.obsm["spatial"], columns=["x", "y"])
    locs.to_csv(f"{output_dir}/locs.tsv", sep="\t", index=False)

    pcs = np.array(adata_filtered.obsm["X_pca"])
    df_pcs = pd.DataFrame(pcs)
    df_pcs.to_csv(f'{output_dir}/pcs.tsv', sep='\t', index=False)

    # sc.pp.neighbors(adata_filtered, n_neighbors=15)
    # sc.tl.leiden(adata_filtered, resolution=.45)
    # df_cluster = pd.DataFrame(np.array(adata_filtered.obs["leiden"]), columns=["leiden_label"])
    # df_cluster.to_csv(f'{output_dir}/leiden.tsv', sep='\t', index=False)

    print(f'===== Exported {dataset} =====')


def plot_marker_gene_expression(args, sample_name, gene_names, dataset="DLPFC", ncol = 5, scale = 0.045, cm = plt.get_cmap("magma"), isCorrGene=False):
    original_spatial = args.spatial
    args.spatial = True
    nrow = int(math.ceil(len(gene_names)/float(ncol)))

    adata = load_DLPFC_data(args, sample_name, v2=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    coord = adata.obsm['spatial'].astype(float) * scale
    x, y = coord[:, 0], coord[:, 1]
    genes = np.char.upper(np.array(adata.var_names).astype(str)) if not isCorrGene else np.array(adata.var_names).astype(str)
    expr = np.asarray(adata.X.todense())
    fig, axs = figure(nrow, ncol, rsz=2.5, csz=2.6, wspace=.2, hspace=.2)
    xlimits = [130, 550]
    ylimits = [100, 530]
    for gid, gene in enumerate(gene_names):
        if gene in genes:
            row = gid // ncol
            col = gid % ncol
            ax = axs[row][col] if nrow > 1 else axs[col]
            ax.axis('off')
            ax.set_xlim(xlimits)
            ax.set_ylim(ylimits)
            ax.invert_yaxis()
            ax.set_title(gene)
            gene_expr = expr[:, genes == gene].flatten()
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

