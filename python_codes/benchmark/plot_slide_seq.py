import argparse
from python_codes.util.config import args
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

def figure(nrow, ncol, rsz=3., csz=3., wspace=.4, hspace=.5, left=None, right=None):
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * csz, nrow * rsz))
    plt_setting()
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right)
    return fig, axs

def plot_annotation(args, adata, nrow = 1, scale = 0.045, ncol=4, rsz=2.5, csz=2.8, wspace=.4, hspace=.5, scatter_sz=1., left=None, right=None):
    fig, axs = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right)
    x, y = adata.obsm["spatial"][:, 0]*scale, adata.obsm["spatial"][:, 1]*scale
    xlim, ylim = None, None
    return fig, axs, x, y, xlim, ylim

def plot_clustering(args, adata, adata_sedr, sample_name, dataset, cm= plt.get_cmap("tab20"), scale=.62, scatter_sz=1., nrow= 1, ncol=4, method="leiden"):
    fig, axs, x, y, xlim, ylim = plot_annotation(args, adata, scale=scale, nrow=nrow, ncol=ncol, rsz=5.0, csz=6.3, wspace=.15, hspace=.2, left=.1, right=.95)
    key_dict = {"seqfish_mouse" : "celltype_mapped_refined", "slideseq_v2": "cluster"}
    annotated_cell_types = adata.obs[key_dict[dataset]].values.astype(str)
    seurat_clusters = pd.read_csv(f"{args.output_dir}/{dataset}/{sample_name}/Seurat/metadata.tsv", sep="\t")["seurat_clusters"].values.flatten().astype(int)
    scanpy_clusters = pd.read_csv(f"{args.output_dir}/{dataset}/{sample_name}/scanpy/{method}.tsv", header=None).values.flatten().astype(int)
    sedr_clusters = pd.read_csv(f"{args.output_dir}/{dataset}/{sample_name}/sedr/{method}.tsv", header=None).values.flatten().astype(int)
    args.spatial = False
    dgi_clusters = pd.read_csv(f"{args.output_dir}/{get_target_fp(args, dataset, sample_name)}/{method}.tsv", header=None).values.flatten().astype(int)
    args.spatial = True
    dgi_sp_clusters = pd.read_csv(f"{args.output_dir}/{get_target_fp(args, dataset, sample_name)}/{method}.tsv",header=None).values.flatten().astype(int)

    clusters_arr = [annotated_cell_types, seurat_clusters, scanpy_clusters, dgi_sp_clusters]#sedr_clusters, dgi_clusters,
    cluster_methods = ["Annotation", "Seurat", "Scanpy", "SpaceFlow"]#, "SEDR", "DGI"
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


def plot_pipeline():
    max_cells = 8000
    args.dataset_dir = f'../../data'
    args.output_dir = f'../../output'
    datasets = ["slideseq_v2"] #, "seqfish_mouse"
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

        plot_clustering(args, adata_filtered, adata_sedr, dataset, dataset, scatter_sz=1.5, scale=1)
        # plot_pseudotime(args, adata_filtered, dataset, dataset, scatter_sz=1.5, scale=1)

if __name__ == "__main__":
    plot_pipeline()