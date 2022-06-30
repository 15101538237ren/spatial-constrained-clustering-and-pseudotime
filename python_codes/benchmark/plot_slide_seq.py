import argparse
import cmcrameri as cmc
from python_codes.util.config import args
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
import warnings
warnings.filterwarnings("ignore")
from python_codes.util.util import *
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial','Roboto']
rcParams['savefig.dpi'] = 300
import matplotlib.pyplot as plt
title_sz = 26

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

def figure(nrow, ncol, rsz=3., csz=3., wspace=.4, hspace=.5, left=None, right=None, bottom=None):
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * csz, nrow * rsz))
    plt_setting()
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom)
    return fig, axs

def plot_annotation(args, adata, nrow = 1, scale = 0.045, ncol=4, rsz=2.5, csz=2.8, wspace=.4, hspace=.5, scatter_sz=1., left=None, right=None, bottom=None):
    fig, axs = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom)
    x, y = adata.obsm["spatial"][:, 0]*scale, adata.obsm["spatial"][:, 1]*scale
    xlim, ylim = None, None
    return fig, axs, x, y, xlim, ylim

def plot_clustering(args, adata, adata_sedr, sample_name, dataset, cm= plt.get_cmap("tab20"), scale=.62, scatter_sz=1., nrow= 1, ncol=4, method="leiden"):
    fig, axs, x, y, xlim, ylim = plot_annotation(args, adata, scale=scale, nrow=nrow, ncol=ncol, rsz=5.0, csz=6.3, wspace=.15, hspace=.2, left=.1, right=.95)
    key_dict = {"seqfish_mouse" : "celltype_mapped_refined", "slideseq_v2": "cluster"}
    annotated_cell_types = adata.obs[key_dict[dataset]].values.astype(str)
    seurat_clusters = pd.read_csv(f"{args.output_dir}/{dataset}/{sample_name}/Seurat/metadata.tsv", sep="\t")["seurat_clusters"].values.flatten().astype(int)
    scanpy_clusters = pd.read_csv(f"{args.output_dir}/{dataset}/{sample_name}/scanpy/{method}.tsv", header=None).values.flatten().astype(int)
    MERINGUE_clusters = pd.read_csv(f"{args.output_dir}/{dataset}/{sample_name}/MERINGUE/metadata.tsv").values.flatten().astype(int)
    sedr_clusters = pd.read_csv(f"{args.output_dir}/{dataset}/{sample_name}/sedr/{method}.tsv", header=None).values.flatten().astype(int)
    args.spatial = False
    dgi_clusters = pd.read_csv(f"{args.output_dir}/{get_target_fp(args, dataset, sample_name)}/{method}.tsv", header=None).values.flatten().astype(int)
    args.spatial = True
    dgi_sp_clusters = pd.read_csv(f"{args.output_dir}/{get_target_fp(args, dataset, sample_name)}/{method}.tsv",header=None).values.flatten().astype(int)

    clusters_arr = [annotated_cell_types, seurat_clusters, MERINGUE_clusters, dgi_sp_clusters]#sedr_clusters, dgi_clusters,
    cluster_methods = ["Annotation", "Seurat", "MERINGUE", "SpaceFlow"]#, "SEDR", "DGI", "Scanpy"
    for cid, clusters in enumerate(clusters_arr):
        row = cid // ncol
        col = cid % ncol
        ax = axs[row][col] if nrow > 1 else axs[col]
        ax.axis('off')
        x = adata.obsm["spatial"][:, 0] if cluster_methods[cid] != "SEDR" else adata_sedr.obsm["spatial"][:, 0]
        y = adata.obsm["spatial"][:, 1] if cluster_methods[cid] != "SEDR" else adata_sedr.obsm["spatial"][:, 1]

        uniq_pred = sorted(np.unique(clusters))
        n_cluster = len(uniq_pred)
        for ccid, cluster in enumerate(uniq_pred):
            color = cm((ccid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
            ind = clusters == cluster
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=str(cluster), marker=".")

        box = ax.get_position()
        height_ratio = 1.0
        ncol_legend = 2 if n_cluster >= 24 else 1
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * height_ratio])
        lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.1,
                         borderaxespad=.1, ncol=ncol_legend, columnspacing=1.0)
        for handle in lgnd.legendHandles:
            handle._sizes = [10]
        ax.set_facecolor("none")
        ax.invert_yaxis()
        ax.set_title(cluster_methods[cid], fontsize=32, pad=10)
    output_dir = f"{args.output_dir}/{dataset}/{sample_name}"
    fig_fp = f"{output_dir}/cluster_comp.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_clustering_of_seurat_with_diff_resolution(args, adata, sample_name, dataset, cm= plt.get_cmap("tab20"), scale=.62, scatter_sz=1., nrow= 1, ncol=4, method="leiden"):
    key_dict = {"seqfish_mouse" : "celltype_mapped_refined", "slideseq_v2": "cluster"}
    resolutions = ["0.4", "0.8", "1.0", "2.0"]
    s1_clusters = pd.read_csv(f"{args.output_dir}/{dataset}/{sample_name}/Seurat/metadata_{resolutions[0]}.tsv", sep="\t")["seurat_clusters"].values.flatten().astype(int)
    s2_clusters = pd.read_csv(f"{args.output_dir}/{dataset}/{sample_name}/Seurat/metadata_{resolutions[1]}.tsv", sep="\t")["seurat_clusters"].values.flatten().astype(int)
    s3_clusters = pd.read_csv(f"{args.output_dir}/{dataset}/{sample_name}/Seurat/metadata_{resolutions[2]}.tsv", sep="\t")["seurat_clusters"].values.flatten().astype(int)
    s4_clusters = pd.read_csv(f"{args.output_dir}/{dataset}/{sample_name}/Seurat/metadata_{resolutions[3]}.tsv", sep="\t")["seurat_clusters"].values.flatten().astype(int)

    clusters_arr = [adata.obs[key_dict[dataset]].values.astype(str), s1_clusters, s2_clusters, s4_clusters] if dataset != "stereo_seq" else [s2_clusters, s3_clusters, s4_clusters]
    cluster_methods = ["Annotation", f"Seurat\n(resolution={resolutions[0]})", f"Seurat\n(resolution={resolutions[1]})", f"Seurat\n(resolution={resolutions[3]})"] if dataset != "stereo_seq" else [f"Seurat\n(resolution={resolutions[1]})", f"Seurat\n(resolution={resolutions[2]})", f"Seurat\n(resolution={resolutions[3]})"]
    csz = 7.8 if dataset != "stereo_seq" else 6.3
    bottom = 0.05
    fig, axs, x, y, xlim, ylim = plot_annotation(args, adata, scale=scale, nrow=nrow, ncol=len(clusters_arr), rsz=6.3, csz=csz, wspace=.15, hspace=.1, left=.1, right=.95, bottom=bottom)
    [x, y] = [x, y] if dataset != "stereo_seq" else [-y, x]
    for cid, clusters in enumerate(clusters_arr):
        row = cid // ncol
        col = cid % ncol
        ax = axs[row][col] if nrow > 1 else axs[col]
        ax.axis('off')

        uniq_pred = sorted(np.unique(clusters))
        n_cluster = len(uniq_pred)
        for ccid, cluster in enumerate(uniq_pred):
            color = cm((ccid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
            ind = clusters == cluster
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=str(cluster), marker=".")

        box = ax.get_position()
        height_ratio = 1.0
        ncol_legend = 2 if n_cluster >= 24 else 1
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * .8])
        lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.1,
                         borderaxespad=.1, ncol=ncol_legend, columnspacing=1.0)
        for handle in lgnd.legendHandles:
            handle._sizes = [10]
        ax.set_facecolor("none")
        ax.invert_yaxis()
        ax.set_title(cluster_methods[cid], fontsize=32, pad=10)
    output_dir = f"{args.output_dir}/{dataset}/{sample_name}"
    fig_fp = f"{output_dir}/cluster_seurat_diff_resolutions.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_clustering_of_diff_subsetting(args, adata, sample_name, dataset, cm= plt.get_cmap("tab20"), scale=.62, scatter_sz=1., nrow= 1, ncol=4, method="leiden"):
    subsettings = ["origin", "subset_1e4", "subset_1e5", "subset_5e5"]
    subsettings_names = ["No Subset", "Random Subset\n#Edges=1e4", "Random Subset\n#Edges=1e5",
                         "Random Subset\n#Edges=5e5"]

    args.spatial = True
    csz = 7.6 if dataset != "stereo_seq" else 6.3
    bottom = 0.05
    fig, axs, x, y, xlim, ylim = plot_annotation(args, adata, scale=scale, nrow=nrow, ncol=len(subsettings), rsz=6.3,
                                                 csz=csz, wspace=.1, hspace=.0, left=.1, right=.95, bottom=bottom)
    [x, y] = [x, y] if dataset != "stereo_seq" else [-y, x]
    for cid, subsetting in enumerate(subsettings):
        row = cid // ncol
        col = cid % ncol
        ax = axs[row][col] if nrow > 1 else axs[col]
        ax.axis('off')
        clusters = pd.read_csv(f"{args.output_dir}/{get_target_fp(args, dataset, sample_name)}/{method}_{subsetting}.tsv",
                                      header=None).values.flatten().astype(int)

        uniq_pred = sorted(np.unique(clusters))
        n_cluster = len(uniq_pred)
        for ccid, cluster in enumerate(uniq_pred):
            color = cm((ccid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
            ind = clusters == cluster
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=str(cluster), marker=".")

        box = ax.get_position()
        height_ratio = 1.0
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * height_ratio])
        lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.1,
                         borderaxespad=.1, ncol=2, columnspacing=1.0)
        for handle in lgnd.legendHandles:
            handle._sizes = [10]
        ax.set_facecolor("none")
        ax.invert_yaxis()
        ax.set_title(subsettings_names[cid], fontsize=36, pad=15)
        box = ax.get_position()
        ax.set_position([box.x0 + (box.x1 - box.x0) * 0.08, box.y0, box.width * 0.8, box.height * 0.76])
    output_dir = f"{args.output_dir}/{dataset}/{sample_name}"
    fig_fp = f"{output_dir}/cluster_diff_subsettings.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_pseudotime_of_diff_subsetting(args, adata, sample_name, dataset, cm = cmc.cm.roma, scale=.62, scatter_sz=1., nrow= 1, ncol=4, method="leiden"):
    subsettings = ["origin", "subset_1e4", "subset_1e5", "subset_5e5"]
    subsettings_names = ["No Subset", "Random Subset\n#Edges=1e4", "Random Subset\n#Edges=1e5",
                         "Random Subset\n#Edges=5e5"]

    args.spatial = True
    bottom = 0.05
    fig, axs, x, y, xlim, ylim = plot_annotation(args, adata, scale=scale, nrow=nrow, ncol=len(subsettings), rsz=6.3,
                                                 csz=7.0, wspace=.15, hspace=.1, left=.1, right=.95, bottom=bottom)
    [x, y] = [x, y] if dataset != "stereo_seq" else [-y, x]
    output_dir = f'{args.output_dir}/{dataset}/{sample_name}/DGI_SP'
    for cid, subsetting in enumerate(subsettings):
        row = cid // ncol
        col = cid % ncol
        ax = axs[row][col] if nrow > 1 else axs[col]
        ax.axis('off')
        ax.invert_yaxis()
        pseudotimes = pd.read_csv(f"{output_dir}/pseudotime_{subsetting}.tsv", header=None).values.flatten().astype(float)

        st = ax.scatter(x, y, s=1, c=pseudotimes, cmap=cm)
        axins = inset_locator.inset_axes(ax, width="5%", height="60%", loc='lower left',
                                         bbox_to_anchor=(1.05, 0.1, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        clb = fig.colorbar(st, cax=axins)
        clb.set_ticks([0.0, np.max(pseudotimes)])
        clb.set_ticklabels(["0", "1"])
        label = "pSM value"
        clb.ax.set_ylabel(label, labelpad=5, rotation=270, fontsize=16, weight='bold')
        ax.set_title(subsettings_names[cid], fontsize=32, pad=15)
        box = ax.get_position()
        ax.set_position([box.x0 + (box.x1 - box.x0) * 0.08, box.y0, box.width * 0.8, box.height * 0.8])
    output_dir = f"{args.output_dir}/{dataset}/{sample_name}"
    fig_fp = f"{output_dir}/pseudotime_diff_subsettings.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')


def plot_pipeline():
    max_cells = 8000
    args.dataset_dir = f'../../data'
    args.output_dir = f'../../output'
    datasets = ["slideseq_v2"]#$"stereo_seq"]#,"stereo_seq" "seqfish_mouse"
    for dataset in datasets:
        print(f'===== Data {dataset} =====')
        data_root = f'{args.dataset_dir}/{dataset}/{dataset}/preprocessed'
        indices_fp = os.path.join(data_root, "indices-for-sedr.npy")
        if os.path.exists(indices_fp):
            adata_filtered, _ = load_preprocessed_data(args, dataset, dataset, sedr=False)
            adata_sedr, _ = load_preprocessed_data(args, dataset, dataset, sedr=True)
        else:
            adata = load_datasets(args, dataset)
            indices = np.random.choice(adata.shape[0], max_cells, replace=False)
            with open(indices_fp, 'wb') as f:
                np.save(f, indices)
            print("Saved indices")
            adata_filtered, spatial_graph = preprocessing_data(args, adata)
            save_preprocessed_data(args, dataset, dataset, adata_filtered, spatial_graph, sedr=False)

            adata = adata[indices, :]
            adata_sedr, spatial_graph = preprocessing_data(args, adata)
            save_preprocessed_data(args, dataset, dataset, adata_sedr, spatial_graph, sedr=True)

        # plot_clustering(args, adata_filtered, adata_sedr, dataset, dataset, scatter_sz=1.5, scale=1)
        # plot_pseudotime(args, adata_filtered, dataset, dataset, scatter_sz=1.5, scale=1)
        # plot_clustering_of_seurat_with_diff_resolution(args, adata_filtered, dataset, dataset, scatter_sz=1.5, scale=1)
        #plot_clustering_of_diff_subsetting(args, adata_filtered, dataset, dataset, scatter_sz=1.5, scale=1)
        plot_pseudotime_of_diff_subsetting(args, adata_filtered, dataset, dataset, scatter_sz=1.5, scale=1)

if __name__ == "__main__":
    plot_pipeline()