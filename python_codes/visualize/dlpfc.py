# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import scanpy as sc
from python_codes.util.util import get_target_fp, load_ST_file
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial','Roboto']

import matplotlib.pyplot as plt

def plt_setting():
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 16
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def figure(nrow, ncol, sz=4):
    plt_setting()
    fig, axs = plt.subplots(1, ncol, figsize=(ncol * sz, nrow * sz))
    plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.2)
    return fig, axs

def plot_clustering(args, sample_name, method="leiden", dataset="DLPFC", HnE=False, cm = plt.get_cmap("plasma"), scale = 0.045):
    ncol = 4 if HnE else 3
    subfig_offset = 1 if HnE else 0

    fig, axs = figure(1, ncol)
    data_root = f'{args.dataset_dir}/{dataset}/{sample_name}'
    adata = load_ST_file(data_root)
    coord = adata.obsm['spatial'].astype(float) * scale
    x, y = coord[:, 0], coord[:, 1]
    df_meta = pd.read_csv(f"{data_root}/metadata.tsv", sep='\t')
    anno_clusters = df_meta['layer_guess'].values.astype(str)
    img = plt.imread(f"{data_root}/spatial/tissue_lowres_image.png")
    limits = [80, 550]
    for ax in axs:
        ax.axis('off')
        ax.imshow(img)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.invert_yaxis()
    ax = axs[subfig_offset]
    ax.set_title("Annotation")
    uniq_annots = np.unique(anno_clusters)
    for cid, cluster in enumerate(uniq_annots):
        if cluster != 'nan':
            color = cm(1. * cid / (len(uniq_annots) + 1))
            ind = anno_clusters == cluster
            ax.scatter(x[ind], y[ind], s=1, color=color, label=cluster)
    ax.legend()

    spatials = [False, True]
    for sid, spatial in enumerate(spatials):
        ax = axs[subfig_offset + sid + 1]
        args.spatial = spatial
        output_dir = f'{args.output_dir}/{get_target_fp(args, dataset, sample_name)}'
        pred_clusters = pd.read_csv(f"{output_dir}/{method}.tsv", header=None).values.flatten().astype(int)
        uniq_pred = np.unique(pred_clusters)
        for cid, cluster in enumerate(uniq_pred):
            color = cm(1. * cid / (len(uniq_pred) + 1))
            ind = pred_clusters == cluster
            ax.scatter(x[ind], y[ind], s=1, color=color, label=cluster)

        title = args.arch if not spatial else "%s + SP" % args.arch
        ax.set_title(title)
    fig_fp = f"{output_dir}/{method}.jpg"
    plt.savefig(fig_fp, dpi=300)
    plt.close('all')