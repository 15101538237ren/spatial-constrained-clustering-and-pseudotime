library(anndata)
library(Giotto)
input_dir = "../data/slideseq_v2/slideseq_v2/export"
ad <- read_h5ad(paste(c(input_dir, "adata.h5ad"), collapse = "/"))

locs <- read.csv(file = paste(c(input_dir, "locs.tsv"), collapse = "/"), sep="\t")

dir.output <- "../output/slideseq_v2/slideseq_v2/DGI_SP/Giotto"
dir.create(dir.output, showWarnings = F)

gt_instructions = createGiottoInstructions(python_path = "~/opt/anaconda3/envs/stlearn/bin/python",
                                           save_dir = dir.output)
gt_obj = createGiottoObject(raw_exprs = ad$X,
                                      spatial_locs = locs,
                                      cell_metadata = ad$var,
                                      gene_metadata = ad$obs,
                                      instructions = gt_instructions)
# preprocessing
gt_obj <- filterGiotto(gobject = gt_obj,
                       expression_values = "raw",
                       expression_threshold = .01, 
                       gene_det_in_min_cells = 10, 
                       min_det_genes_per_cell = 0)
gt_obj <- normalizeGiotto(gobject = gt_obj)

# dimension reduction
gt_obj <- calculateHVG(gobject = gt_obj)
gt_obj <- runPCA(gobject = gt_obj)
gt_obj <- runUMAP(gt_obj, dimensions_to_use = 1:5)

# leiden
gt_obj = doLeidenCluster(gt_obj, name = 'leiden_clus', nn_network_to_use='knn', network_name="knn.pca")
gt_obj = doLouvainCluster(gt_obj, name = 'louvain_clus', nn_network_to_use='knn', network_name="knn.pca")
gt_obj = doKmeans(gt_obj, centers = 20, name = 'kmeans_clus')

# calculate cluster similarities to see how individual clusters are correlated
cluster_similarities = getClusterSimilarity(gt_obj,
                                            cluster_column = 'leiden_clus')

# merge similar clusters based on correlation and size parameters
gt_obj = mergeClusters(gt_obj, cluster_column = 'leiden_clus',
                       min_cor_score = 0.7, force_min_group_size = 4)

# visualize
pDataDT(gt_obj)
plotUMAP_2D(gt_obj, cell_color = 'merged_cluster', point_size = 3)



