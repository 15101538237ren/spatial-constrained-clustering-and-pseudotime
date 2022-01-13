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

def plot_clustering(args, adata, sample_name, dataset, cm= plt.get_cmap("tab20"), scale=.62, scatter_sz=1., nrow= 1):
    fig, ax, x, y, xlim, ylim = plot_annotation(args, adata, sample_name, scale=scale, nrow=nrow, ncol=1, rsz=5, csz=6., wspace=.3, hspace=.1)
    output_dir = f'{args.output_dir}/{dataset}/{sample_name}/Seurat'
    pred_clusters = pd.read_csv(f"{output_dir}/metadata.tsv", sep="\t")["seurat_clusters"].values.flatten().astype(int)
    uniq_pred = np.unique(pred_clusters)
    n_cluster = len(uniq_pred)
    for cid, cluster in enumerate(uniq_pred):
        color = cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
        ind = pred_clusters == cluster
        if dataset == "stereo_seq":
            ax.scatter(-y[ind], x[ind], s=scatter_sz, color=color, label=cluster, marker=".")
        else:
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster, marker=".")

    box = ax.get_position()
    height_ratio = 1.0
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * height_ratio])
    lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.1,
                     borderaxespad=.1, ncol=2, columnspacing=1.0)
    for handle in lgnd.legendHandles:
        handle._sizes = [8]
    ax.set_facecolor("none")
    ax.invert_yaxis()
    fig_fp = f"{output_dir}/seurat.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_pipeline():
    args.dataset_dir = f'../../data'
    args.output_dir = f'../../output'
    datasets = ["stereo_seq"]#, , "seqfish_mouse""slideseq_v2"
    for did, dataset in enumerate(datasets):
        print(f'===== Data {dataset} =====')
        data_root = f'{args.dataset_dir}/{dataset}/{dataset}/preprocessed'
        if os.path.exists(f"{data_root}/adata.h5ad"):
            adata_filtered, spatial_graph = load_preprocessed_data(args, dataset, dataset)
        else:
            adata = load_datasets(args, dataset)
            adata_filtered, spatial_graph = preprocessing_data(args, adata)
            save_preprocessed_data(args, dataset, dataset, adata_filtered, spatial_graph)
        plot_clustering(args, adata_filtered, dataset, dataset, scatter_sz=1.5, scale=1)

if __name__ == "__main__":
    plot_pipeline()