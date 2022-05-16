import argparse
from scipy.spatial import distance_matrix
from python_codes.util.config import args
from python_codes.sedr.graph_func import graph_construction
from python_codes.sedr.utils_func import mk_dir, adata_preprocess
from python_codes.sedr.SEDR_train import SEDR_Train
import warnings
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

def plot_annotation(args, adata, sample_name, nrow = 1, scale = 0.045, ncol=4, rsz=2.5, csz=2.8, wspace=.4, hspace=.5, scatter_sz=1, left=None, right=None):
    fig, ax = figure(nrow, ncol, rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right)
    ax.axis('off')
    x, y = adata.obsm["spatial"][:, 0]*scale, adata.obsm["spatial"][:, 1]*scale
    xlim, ylim = None, None
    return fig, ax, x, y, xlim, ylim

def get_params():
    # ################ Parameter setting
    warnings.filterwarnings('ignore')
    torch.cuda.cudnn_enabled = False
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda:6' if torch.cuda.is_available() else 'cpu'
    print('===== Using device: ' + device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=15, help='parameter k in spatial graph')
    parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                        help='graph distance type: euclidean/cosine/correlation')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--cell_feat_dim', type=int, default=200, help='Dim of PCA')
    parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
    parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
    parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
    parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
    parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
    parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
    parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
    parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
    parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
    parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
    parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
    parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
    parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
    parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
    parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.')

    params = parser.parse_args()
    params.device = device
    return params

def res_search_fixed_clus(clustering_method, adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=False):
        if clustering_method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs[clustering_method]).leiden.unique())
        else:
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(np.unique(pd.DataFrame(adata.obs[clustering_method].cat.codes.values).values.flatten()))
        print("Try resolution %3f found %d clusters: target %d" % (res, count_unique, fixed_clus_count))
        if count_unique == fixed_clus_count:
            print("Found resolution:" + str(res))
            return res
        elif count_unique > fixed_clus_count:
            print("Found resolution: %.3f" % (res - increment))
            return res - increment

def plot_clustering(args, adata, sample_name, dataset, method="leiden", cm= plt.get_cmap("tab20"), scale=.62, scatter_sz=1., nrow= 1):
    fig, ax, x, y, xlim, ylim = plot_annotation(args, adata, sample_name, scale=scale, nrow=nrow, ncol=1, rsz=5, csz=6, wspace=.1, hspace=.1, left=.1, right=.95)
    output_dir = f'{args.output_dir}/{dataset}/{sample_name}/sedr'
    pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(int)
    uniq_pred = np.unique(pred_clusters)
    n_cluster = len(uniq_pred)
    for cid, cluster in enumerate(uniq_pred):
        color = cm((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
        ind = pred_clusters == cluster
        if dataset == "stereo_seq":
            ax.scatter(-y[ind], x[ind], s=scatter_sz, color=color, label=cluster, marker=".")
        else:
            ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster, marker=".")
    ax.set_facecolor("none")
    # title = "SEDR"
    # ax.set_title(title, fontsize=title_sz, pad=-30)
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

def plot_pseudotime(args, adata, sample_name, dataset, cm = plt.get_cmap("gist_rainbow"), scale = 0.62, scatter_sz=1.3, nrow = 1):
    fig, ax, x, y, xlim, ylim = plot_annotation(args, adata, sample_name, scale=scale, nrow=nrow, ncol=1, rsz=5,
                                                csz=5.5, wspace=.3, hspace=.4)
    output_dir = f'{args.output_dir}/{dataset}/{sample_name}/sedr'
    pseudotimes = pd.read_csv(f"{output_dir}/pseudotime.tsv", header=None).values.flatten().astype(float)
    st = ax.scatter(x, y, s=scatter_sz, c=pseudotimes, cmap=cm, marker=".")
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = fig.colorbar(st, cax=cax)
    clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
    title = "SEDR"
    ax.set_title(title, fontsize=title_sz)
    ax.set_facecolor("none")
    fig_fp = f"{output_dir}/psudotime.pdf"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')

def plot_pipeline():
    max_cells = 10000
    params = get_params()
    args.dataset_dir = f'../../data'
    args.output_dir = f'../../output'
    datasets = ["stereo_seq"] #"slideseq_v2", "seqfish_mouse",
    for did, dataset in enumerate(datasets):
        print(f'===== Data {dataset} =====')
        data_root = f'{args.dataset_dir}/{dataset}/{dataset}/preprocessed'
        indices_fp = os.path.join(data_root, "indices-for-sedr.npy")
        if os.path.exists(indices_fp):
            with open(indices_fp, 'rb') as f:
                indices = np.load(f)
                print("loaded indices successful!")
            adata_filtered, spatial_graph = load_preprocessed_data(args, dataset, dataset, sedr=True)
        else:
            adata = load_datasets(args, dataset)
            indices = np.random.choice(adata.shape[0], max_cells, replace=False)
            with open(indices_fp, 'wb') as f:
                np.save(f, indices)
            print("Saved indices")
            adata = adata[indices, :]
            adata_filtered, spatial_graph = preprocessing_data(args, adata)
            sc.pp.pca(adata_filtered, n_comps=params.cell_feat_dim)
            save_preprocessed_data(args, dataset, dataset, adata_filtered, spatial_graph, sedr=True)

        plot_clustering(args, adata_filtered, dataset, dataset, scatter_sz=1.5, scale=1)
        plot_pseudotime(args, adata_filtered, dataset, dataset, scatter_sz=1.5, scale=1)

def basic_pipeline(subset=False):
    params = get_params()
    args.dataset_dir = f'../../data'
    args.output_dir = f'../../output'
    max_cells = 10000
    datasets = ["stereo_seq"]#"slideseq_v2", "seqfish_mouse",
    n_neighbors = [15, 15, 15]
    resolutions = [1.0, 0.8, 0.8]
    for did, dataset in enumerate(datasets):
        print(f'===== Data {dataset} =====')
        data_root = f'{args.dataset_dir}/{dataset}/{dataset}/preprocessed'
        if subset:
            indices_fp = os.path.join(data_root, "indices-for-sedr.npy")
            if os.path.exists(indices_fp):
                with open(indices_fp, 'rb') as f:
                    indices = np.load(f)
                    print("loaded indices successful!")
                adata_filtered = load_preprocessed_data(args, dataset, dataset, sedr=True)
            else:
                adata = load_datasets(args, dataset)
                indices = np.random.choice(adata.shape[0], max_cells, replace=False)
                with open(indices_fp, 'wb') as f:
                    np.save(f, indices)
                print("Saved indices")
                adata = adata[indices, :]
                adata_filtered = preprocessing_data_sedr(args, adata, pca_n_comps=params.cell_feat_dim)
                save_preprocessed_data(args, dataset, dataset, adata_filtered, None, sedr=True)
        else:
            adata = load_datasets(args, dataset)
            adata_filtered = preprocessing_data_sedr(args, adata, pca_n_comps=params.cell_feat_dim)
        graph_dict = graph_construction(adata_filtered.obsm['spatial'], adata_filtered.shape[0], params)
        print('==== Graph Construction Finished')
        params.save_path = f'{args.output_dir}/{dataset}/{dataset}/sedr'
        mk_dir(params.save_path)
        params.cell_num = adata_filtered.shape[0]
        print('==== Graph Construction Finished')

        # ################## Model training
        adata_pca = adata_filtered.obsm['X_pca']
        sedr_net = SEDR_Train(adata_pca, graph_dict, params)
        if params.using_dec:
            sedr_net.train_with_dec()
        else:
            sedr_net.train_without_dec()
        embeddings, _, _, _ = sedr_net.process()

        np.savez(f'{params.save_path}/sedr_embedding.npz', embeddings=embeddings, params=params)
        embeddings = np.load(f'{params.save_path}/sedr_embedding.npz')["embeddings"]
        # ################## Result plot
        adata = anndata.AnnData(embeddings)
        adata.uns['spatial'] = adata_filtered.obsm['spatial']
        adata.obsm['spatial'] = adata_filtered.obsm['spatial']

        sc.pp.neighbors(adata, n_neighbors=n_neighbors[did])
        sc.tl.umap(adata)
        resolution = res_search_fixed_clus("leiden", adata, 8) if dataset == "stereo_seq" else resolutions[did]
        sc.tl.leiden(adata, resolution=resolution)
        sc.tl.paga(adata)
        df_meta = pd.DataFrame(np.array(adata.obs['leiden']))
        df_meta.to_csv(f'{params.save_path}/leiden.tsv', sep='\t', header=False, index=False)

        indices = np.arange(adata.shape[0])
        selected_ind = np.random.choice(indices, 5000, False)
        sub_adata_x = adata.X[selected_ind, :]
        sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
        adata.uns['iroot'] = np.argmax(-sum_dists)
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata)
        pseudotimes = adata.obs['dpt_pseudotime'].to_numpy()
        pseudotime_fp = f'{params.save_path}/pseudotime.tsv'
        np.savetxt(pseudotime_fp, pseudotimes, fmt='%.5f', header='', footer='', comments='')
        print("Saved %s succesful!" % pseudotime_fp)

if __name__ == "__main__":
    basic_pipeline()
    #plot_pipeline()
