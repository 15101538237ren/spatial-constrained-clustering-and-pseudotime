# -*- coding: utf-8 -*-
import os, math, shutil
import cmcrameri as cmc
#from python_codes.train.train import train
from python_codes.train.clustering import clustering
from python_codes.train.pseudotime import pseudotime
from python_codes.util.util import load_slideseqv2_data, preprocessing_data, save_preprocessed_data, load_preprocessed_data, save_features
import warnings
from scipy.sparse import csr_matrix
from python_codes.train.clustering import clustering
from python_codes.train.pseudotime import pseudotime
from scipy.spatial import distance_matrix
warnings.filterwarnings("ignore")
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

####################################
#----------Get Annotations---------#
####################################

def get_clusters(args, dataset, sample_name, method="leiden"):
    original_spatial = args.spatial
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(str)
    args.spatial = original_spatial
    return pred_clusters

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

def figure(nrow, ncol, rsz=3., csz=3., wspace=.4, hspace=.5, left=None, right=None):
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * csz, nrow * rsz))
    plt_setting()
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right)
    return fig, axs

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

def plot_annotation(args, adata, sample_name, nrow = 1, scale = 0.045, ncol=4, rsz=2.5, csz=2.8, wspace=.4, hspace=.5, scatter_sz=1.):
    fig, axs = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace)
    if nrow == 1:
        for ax in axs:
            ax.axis('off')
    ax = axs[0]
    x, y = adata.obsm["spatial"][:, 0]*scale, adata.obsm["spatial"][:, 1]*scale
    annotated_cell_types = adata.obs["cluster"]
    cell_type_strs = annotated_cell_types.cat.categories.astype(str)
    cell_type_ints = annotated_cell_types.values.codes
    cell_type_colors = list(adata.uns['cluster_colors'].astype(str))
    colors = np.array([cell_type_colors[item] for item in cell_type_ints])
    for cid in range(len(cell_type_colors)):
        cit = cell_type_ints == cid
        ax.scatter(x[cit], y[cit], s=scatter_sz, c=colors[cit], label=cell_type_strs[cid], marker=".")
    ax.set_facecolor("none")
    ax.set_title("Annotation", fontsize=title_sz)
    xlim, ylim = None, None
    ax.invert_yaxis()
    return fig, axs, x, y, xlim, ylim


def plot_clustering(args, adata, sample_name, method="leiden", dataset="slideseq_v2", cm = plt.get_cmap("tab20"), scale=.62, scatter_sz=1., nrow= 1):
    original_spatial = args.spatial
    fig, axs, x, y, xlim, ylim = plot_annotation(args, adata, sample_name, scale=scale, nrow=nrow, ncol=3, rsz=5, csz=5.5, wspace=.3, hspace=.4)
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
        ax.invert_yaxis()
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

def plot_pseudotime(args, adata, sample_name, dataset="slideseq_v2", cm = plt.get_cmap("gist_rainbow"), scale = 0.62, scatter_sz=1.3, nrow = 1):
    original_spatial = args.spatial
    fig, axs, x, y, _, _ = plot_annotation(args, adata, sample_name, scale=scale, nrow=nrow, ncol=3, rsz=5, csz=5.5, wspace=.3, hspace=.4)
    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        ax = axs[sid + 1]
        args.spatial = spatial
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
        pseudotimes = pd.read_csv(f"{output_dir}/pseudotime.tsv", header=None).values.flatten().astype(float)
        st = ax.scatter(x, y, s=scatter_sz, c=pseudotimes, cmap=cm, marker=".")
        ax.invert_yaxis()
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

def plot_expr_in_ST(args, adata, genes, sample_name, dataset, scatter_sz= 1., cm = plt.get_cmap("RdPu"), n_cols = 4, max_expr_threshold=.0):
    args.spatial = True
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    mkdir(output_dir)
    n_genes = len(genes)
    n_rows = int(math.ceil(n_genes/n_cols))
    fig, axs = figure(n_rows, n_cols, rsz=5.5, csz=5.2, wspace=.1, hspace=.3, left=.05, right=.95)
    exprs = np.array(adata.X.todense()).astype(float)
    all_genes = np.array(adata.var_names)

    x, y = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
    for gid, gene in enumerate(genes):
        row = gid // n_cols
        col = gid % n_cols
        ax = axs[row][col] if n_rows > 1 else axs[col]
        expr = exprs[:, all_genes == gene]
        expr = (expr - expr.mean())/expr.std()
        ax = set_ax_for_expr_plotting(ax)
        st = ax.scatter(x, y, s=scatter_sz, c=expr, cmap=cm, vmin=0, vmax=6)
        # if gid == len(genes) - 1:
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     clb = fig.colorbar(st, cax=cax)
        #     clb.ax.set_ylabel("Expr.", labelpad=10, rotation=270, fontsize=10, weight='bold')
        ax.set_title(gene, fontsize=30, pad=10)
    fig_fp = f"{output_dir}/ST_expression.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_umap_comparison_with_coord_alpha(args, sample_name, dataset, n_neighbors=15):
    methods = ["scanpy",  "Seurat", "DGI", "DGI_SP"]
    files = ["PCA.tsv", "seurat.PCs.tsv", "features.tsv", "features.tsv"]
    nrow, ncol = 1, len(methods)

    data_root = f'{args.dataset_dir}/{dataset}/{dataset}/preprocessed'
    if os.path.exists(f"{data_root}/adata.h5ad"):
        adata_filtered, spatial_graph = load_preprocessed_data(args, dataset, dataset)
    else:
        adata = load_stereo_seq_data(args)
        adata_filtered, spatial_graph = preprocessing_data(args, adata)
        save_preprocessed_data(args, dataset, dataset, adata_filtered, spatial_graph)

    coord = adata_filtered.obsm['spatial'].astype(float)
    x, y = coord[:, 0], coord[:, 1]
    normed_x = (x - np.min(x))/(np.max(x) - np.min(x))
    normed_y = (y - np.min(y))/(np.max(y) - np.min(y))
    normed_c = np.sqrt(normed_x**2 + normed_y**2)
    normed_c = (normed_c - np.min(normed_c))/(np.max(normed_c) - np.min(normed_c))

    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}'
    fig, axs = figure(nrow, ncol, rsz=5.5, csz=6., wspace=.1, hspace=.1, left=.05, right=.95)
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
            adata = sc.read_csv(feature_fp, delimiter="\t")
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
            sc.tl.umap(adata)
            umap_positions = adata.obsm["X_umap"]
            np.savetxt(umap_positions_fp, umap_positions, fmt='%.5f\t%.5f', header='', footer='', comments='')
        else:
            umap_positions = pd.read_csv(umap_positions_fp, header=None, sep="\t").values.astype(float)

        if method != "Seurat":
            pred_clusters = pd.read_csv(f"{output_dir}/leiden.tsv", header=None).values.flatten().astype(int)
        else:
            pred_clusters = pd.read_csv(f"{output_dir}/metadata.tsv", sep="\t")["seurat_clusters"].values.flatten().astype(int)
        cluster_names = list(np.unique(pred_clusters))
        n_cluster = len(cluster_names)
        cm = plt.get_cmap("tab20")
        for cid, cluster in enumerate(cluster_names):
            ind = pred_clusters == cluster
            umap_sub = umap_positions[ind]
            alphas = normed_c[ind]
            color = to_hex(cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster))
            color_gradients = linear_gradient(color, n=6)["hex"]
            n = umap_sub.shape[0]
            colors = np.array([color_gradients[int(alphas[i] // 0.2) + 1] for i in range(n)])
            ax.scatter(umap_sub[:, 0], umap_sub[:, 1], s=1, color=colors, label=cluster)
        if mid == len(methods) - 1:
            box = ax.get_position()
            height_ratio = 1.0
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * height_ratio])
            ax.legend(loc='center left', fontsize='x-small', bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.05,
                      borderaxespad=.1)
        ax.set_title(method.replace("_", " + "), fontsize=title_sz)
    fig_fp = f"{output_dir}/umap_comparison-calpha.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')


def plot_rank_marker_genes_group(args, dataset, sample_name, adata_filtered, method="cluster", top_n_genes=3):
    original_spatial = args.spatial
    args.spatial = True
    pred_clusters = get_clusters(args, dataset, sample_name)
    output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
    adata_filtered.obs[method] = pd.Categorical(pred_clusters)
    sc.tl.rank_genes_groups(adata_filtered, method, method='wilcoxon')
    # sc.pl.rank_genes_groups(adata_filtered, n_genes=25, ncols=5, fontsize=10, sharey=False, save=f"{sample_name}_ranks_gby_{method}.pdf")
    # sc.pl.rank_genes_groups_heatmap(adata_filtered, n_genes=top_n_genes, standard_scale='var',  show_gene_labels=True, save=f"{sample_name}_heatmap_gby_{method}.pdf")
    sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=top_n_genes, standard_scale='var', cmap='bwr', save=f"{sample_name}_mean_expr_gby_{method}.pdf")
    # sc.pl.rank_genes_groups_dotplot(adata_filtered, n_genes=top_n_genes, values_to_plot="logfoldchanges", cmap='bwr', vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change', save=f"{sample_name}_dot_lfc_gby_{method}.pdf")
    # sc.pl.rank_genes_groups_matrixplot(adata_filtered, n_genes=top_n_genes, values_to_plot="logfoldchanges", cmap='bwr', vmin=-4, vmax=4, min_logfoldchange=1.5, colorbar_title='log fold change', save=f"{sample_name}_matrix_lfc_gby_{method}.pdf")
    # sc.pl.rank_genes_groups_matrixplot(adata_filtered, n_genes=top_n_genes, cmap='bwr', colorbar_title='Mean Expr.', save=f"{sample_name}_matrix_mean_expr_gby_{method}.pdf")

    files = [
             # f"rank_genes_groups_cluster{sample_name}_ranks_gby_{method}.pdf",
             # f"heatmap{sample_name}_heatmap_gby_{method}.pdf",
             f"dotplot_{sample_name}_mean_expr_gby_{method}.pdf",
             # f"dotplot_{sample_name}_dot_lfc_gby_{method}.pdf",
             # f"matrixplot_{sample_name}_matrix_lfc_gby_{method}.pdf"#,
             # f"matrixplot_{sample_name}_matrix_mean_expr_gby_{method}.pdf"
    ]

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

def plot_pseudotime_comparison(args, adata, sample_name, dataset="slideseq_v2", cm =cmc.cm.roma, scale = 0.045, n_neighbors=50, root_cell_type = None, cell_types=None):
    methods = ["Seurat", "monocle", "slingshot", "DGI_SP"]#, "stLearn", "DGI"
    files = ["seurat.PCs.tsv", None, None, "features_origin.tsv"]#, "PCs.tsv", "features.tsv"
    nrow, ncol = 1, len(methods)

    coord = adata.obsm['spatial'].astype(float) * scale
    x, y = coord[:, 0], coord[:, 1]

    fig, axs = figure(nrow, ncol, rsz=5.0, csz=6.2, wspace=.35, hspace=.3)
    for ax in axs:
        ax.axis('off')
        ax.invert_yaxis()

    for mid, method in enumerate(methods):
        print(f"Processing {sample_name} {method}")
        col = mid % ncol
        ax = axs[col]
        output_dir = f'{args.output_dir}/{dataset}/{sample_name}/{method}'
        fn = "pseudotime.tsv" if method != "DGI_SP" else "pseudotime_origin.tsv"
        pseudotime_fp = f"{output_dir}/{fn}"
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


####################################
#-------------Pipelines------------#
####################################

def export_data_pipeline(args):
    dataset = "slideseq_v2"
    output_dir = f'{args.dataset_dir}/{dataset}/{dataset}/export'
    mkdir(output_dir)

    adata = load_slideseqv2_data()
    adata_filtered, spatial_graph = preprocessing_data(args, adata)
    write_exchangeable_loom(adata_filtered,f'{output_dir}/adata_filtered.loom')

    locs = pd.DataFrame(adata_filtered.obsm["spatial"], columns=["x", "y"])
    locs.to_csv(f"{output_dir}/locs.tsv", sep="\t", index=False)

    pcs = np.array(adata_filtered.obsm["X_pca"])
    df_pcs = pd.DataFrame(pcs)
    df_pcs.to_csv(f'{output_dir}/pcs.tsv', sep='\t', index=False)

    sc.pp.neighbors(adata_filtered, n_neighbors=30)
    sc.tl.leiden(adata_filtered, resolution=.3)

    df_cluster = pd.DataFrame(np.array(adata_filtered.obs["leiden"]), columns=["leiden_label"])
    df_cluster.to_csv(f'{output_dir}/leiden.tsv', sep='\t', index=False)

def train_pipeline(args, adata_filtered, spatial_graph, sample_name, dataset="slideseq_v2", clustering_method="leiden", resolution=1., n_neighbors=15, isTrain=True):
    for spatial in [True]:#False,
        args.spatial = spatial
        if isTrain:
            embedding = train(args, adata_filtered, spatial_graph)
            save_features(args, embedding, dataset, sample_name)
        clustering(args, dataset, sample_name, clustering_method, n_neighbors=n_neighbors, resolution=resolution)
        pseudotime(args, dataset, sample_name, n_neighbors=n_neighbors, resolution=resolution)

def basic_pipeline(args):
    dataset = "slideseq_v2"
    clustering_method = "leiden"
    sample_list = ['slideseq_v2']

    for sample_idx, sample_name in enumerate(sample_list):
        print(f'===== Data {sample_idx + 1} : {sample_name}')
        data_root = f'{args.dataset_dir}/{dataset}/{sample_name}/preprocessed'
        if os.path.exists(f"{data_root}/adata.h5ad"):
            adata_filtered, spatial_graph = load_preprocessed_data(args, dataset, sample_name)
        else:
            adata = load_slideseqv2_data()
            adata_filtered, spatial_graph = preprocessing_data(args, adata)
            save_preprocessed_data(args, dataset, sample_name, adata_filtered, spatial_graph)

        # train_pipeline(args, adata_filtered, spatial_graph, sample_name, n_neighbors=8, isTrain=True)
        # plot_clustering(args, adata_filtered, sample_name, scatter_sz=1.5, scale=1, method=clustering_method)
        # plot_pseudotime(args, adata_filtered, sample_name, scatter_sz=1.5, scale=1)
        # plot_rank_marker_genes_group(args, sample_name, adata_filtered, top_n_genes=5)
        # get_correlation_matrix_btw_clusters(args, sample_name, adata_filtered)
        #plot_umap_comparison_with_coord_alpha(args, dataset, dataset)
        plot_pseudotime_comparison(args, adata_filtered, dataset, dataset)

def expr_analysis_pipeline(args):
    dataset = "slideseq_v2"
    genes = ["Atp2b1", "Chgb", "Lrrtm4", "Enpp2", "Mbp", "Pcp4", "Ptgds", "Meg3"]#, "Necab2", "Ncdn"
    print(f'===== Data: {dataset} =====')
    adata = load_slideseqv2_data()
    plot_expr_in_ST(args, adata, genes, dataset, dataset, scatter_sz=2.)

def marker_gene_pipeline(args):
    dataset = "slideseq_v2"

    print(f'===== Data: {dataset} =====')
    data_root = f'{args.dataset_dir}/{dataset}/{dataset}/preprocessed'
    if os.path.exists(f"{data_root}/adata.h5ad"):
        adata_filtered, spatial_graph = load_preprocessed_data(args, dataset, dataset)
    else:
        adata = load_slideseqv2_data()
        adata_filtered, spatial_graph = preprocessing_data(args, adata)
        save_preprocessed_data(args, dataset, dataset, adata_filtered, spatial_graph)
    adata_filtered.var_names = np.char.upper(np.array(adata_filtered.var_names).astype(str))
    plot_rank_marker_genes_group(args, dataset, dataset, adata_filtered, top_n_genes=3)