library(mclust)
library(ggplot2)
library(patchwork)
library(Seurat)
library(mclust)
options(bitmapType = 'cairo')

args <- commandArgs(trailingOnly = TRUE)
sample <- args[1]

sp_data <- readRDS(file.path('../output/DLPFC/', sample, '/Seurat/Seurat_final.rds'))

##### SpatialDimPlot
metadata <- read.table(file.path('../data/DLPFC/', sample, 'metadata.tsv'), sep='\t', header=TRUE)

sedr_cluster <- read.table(file.path('../output/DLPFC/', sample, 'SEDR/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
sedr_cluster$sed_labels <- sedr_cluster$leiden_fixed_clusCount

seurat_cluster <- read.table(file.path('../output/DLPFC/', sample, '/Seurat/metadata.tsv'), sep='\t', header=TRUE)
spaGCN_cluster <- read.table(file.path('../output/DLPFC/', sample, '/SpaGCN/metadata.tsv'), sep='\t', header=TRUE)
BayesSpace_cluster <- read.table(file.path('../output/DLPFC/', sample, '/BayesSpace/metadata.tsv'), sep='\t', header=TRUE)
Giotto_cluster <- read.table(file.path('../output/DLPFC/', sample, '/Giotto/metadata.tsv'), sep='\t', header=TRUE)
row.names(Giotto_cluster) <- Giotto_cluster$cell_ID
stLearn_cluster <- read.table(file.path('../output/DLPFC/', sample, '/stLearn/metadata.tsv'), sep='\t', header=TRUE)

# APPNP_cluster <- read.table(file.path('../output/DLPFC/', sample, 'APPNP_zdim50_knn10_2z1z_nepoch_150/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
# GAT_cluster <- read.table(file.path('../output/DLPFC/', sample, 'GAT_zdim50_knn10_2z1z_nepoch_150/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
# RGCN_cluster <- read.table(file.path('../output/DLPFC/', sample, 'RGCN_zdim50_knn10_2z1z_nepoch_150/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
# dgi_cluster <- read.table(file.path('../output/DLPFC/', sample, 'GCN_zdim50_knn10_2z1z_nepoch_150_weights1/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
# dgi_expr_cluster <- read.table(file.path('../output/DLPFC/', sample, 'GCN_zdim50_knn10_2z1z_nepoch_150_just_expr_weights/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
# dgi_spatial_cluster <- read.table(file.path('../output/DLPFC/', sample, 'GCN_zdim50_knn10_2z1z_nepoch_150_just_spatial_weights/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
# dgi_expr_spatial_cluster <- read.table(file.path('../output/DLPFC/', sample, 'GCN_zdim50_knn10_2z1z_nepoch_150/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
# vasc_cluster <- read.table(file.path('../output/DLPFC/', sample, 'VASC/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
# vasc_sp_cluster <- read.table(file.path('../output/DLPFC/', sample, 'VASC_with_spatial/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
# gae_cluster <- read.table(file.path('../output/DLPFC/', sample, 'GAE/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
# gae_sp_cluster <- read.table(file.path('../output/DLPFC/', sample, 'GAE_with_spatial/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
# vgae_cluster <- read.table(file.path('../output/DLPFC/', sample, 'VGAE/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
# vgae_sp_cluster <- read.table(file.path('../output/DLPFC/', sample, 'VGAE_with_spatial/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
dgi_w_cluster <- read.table(file.path('../output/DLPFC/', sample, 'DGI/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
dgi_sp_cluster <- read.table(file.path('../output/DLPFC/', sample, 'DGI_with_spatial/metadata.tsv'), sep='\t', header=TRUE, row.names =1)

truth <- as.factor(metadata$layer_guess)
truth <- factor(truth, levels=c('WM', 'nan', 'Layer6', 'Layer5', 'Layer4', 'Layer3', 'Layer2', 'Layer1'))
sp_data <- AddMetaData(sp_data, truth, col.name = 'layer_guess')
sp_data <- AddMetaData(sp_data, seurat_cluster$seurat_clusters, col.name = 'Seurat')
sp_data <- AddMetaData(sp_data, spaGCN_cluster$refined_pred, col.name = 'SpaGCN')
sp_data <- AddMetaData(sp_data, BayesSpace_cluster$spatial.cluster, col.name = 'BayesSpace')
sp_data <- AddMetaData(sp_data, Giotto_cluster[, 'HMRF_cluster', drop=F], col.name = 'Giotto')
sp_data <- AddMetaData(sp_data, stLearn_cluster$X_pca_kmeans, col.name = 'stLearn')
sp_data <- AddMetaData(sp_data, sedr_cluster$SEDR, col.name = 'SEDR')
# sp_data <- AddMetaData(sp_data, APPNP_cluster$APPNP_zdim50_knn10_2z1z_nepoch_150, col.name = 'APPNP')
# sp_data <- AddMetaData(sp_data, GAT_cluster$GAT_zdim50_knn10_2z1z_nepoch_150, col.name = 'GAT')
# sp_data <- AddMetaData(sp_data, RGCN_cluster$RGCN_zdim50_knn10_2z1z_nepoch_150, col.name = 'RGCN')
# sp_data <- AddMetaData(sp_data, dgi_cluster$GCN_zdim50_knn10_2z1z_nepoch_150_weights1, col.name = 'DGI_spatial_weights_1')
# sp_data <- AddMetaData(sp_data, dgi_expr_cluster$GCN_zdim50_knn10_2z1z_nepoch_150_just_expr_weights, col.name = 'DGI_expr_weights')
# sp_data <- AddMetaData(sp_data, dgi_spatial_cluster$GCN_zdim50_knn10_2z1z_nepoch_150_just_spatial_weights, col.name = 'DGI_spatial_weights')
# sp_data <- AddMetaData(sp_data, dgi_expr_spatial_cluster$GCN_zdim50_knn10_2z1z_nepoch_150, col.name = 'DGI_expr_spatial_weights')
# sp_data <- AddMetaData(sp_data, vasc_cluster$VASC, col.name = 'VASC')
# sp_data <- AddMetaData(sp_data, vasc_sp_cluster$VASC_with_spatial, col.name = 'VASC_with_spatial')
# sp_data <- AddMetaData(sp_data, gae_cluster$GAE, col.name = 'GAE')
# sp_data <- AddMetaData(sp_data, gae_sp_cluster$GAE_with_spatial, col.name = 'GAE_with_spatial')
# sp_data <- AddMetaData(sp_data, vgae_cluster$VGAE, col.name = 'VGAE')
# sp_data <- AddMetaData(sp_data, vgae_sp_cluster$VGAE_with_spatial, col.name = 'VGAE_with_spatial')
sp_data <- AddMetaData(sp_data, dgi_w_cluster$DGI, col.name = 'DGI')
sp_data <- AddMetaData(sp_data, dgi_sp_cluster$DGI_with_spatial, col.name = 'DGI_with_spatial')

Seurat_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$Seurat)
SpaGCN_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$SpaGCN)
BayesSpace_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$BayesSpace)
Giotto_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$Giotto)
stLearn_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$stLearn)
SEDR_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$SEDR)
# APPNP_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$APPNP)
# GAT_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$GAT)
# RGCN_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$RGCN)
# DGI_w1_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$DGI_spatial_weights_1)
# DGI_expr_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$DGI_expr_weights)
# DGI_spatial_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$DGI_spatial_weights)
# DGI_expr_spatial_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$DGI_expr_spatial_weights)
# VASC_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$VASC)
# VASC_spatial_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$VASC_with_spatial)
# GAE_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$GAE)
# GAE_spatial_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$GAE_with_spatial)
# VGAE_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$VGAE)
# VGAE_spatial_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$VGAE_with_spatial)
DGI_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$DGI)
DGI_spatial_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$DGI_with_spatial)

df_clusters <- data.frame(layer_guess = sp_data@meta.data$layer_guess,
                          # DGI_expr_spatial = as.factor(sp_data@meta.data$DGI_expr_spatial_weights),
                          # DGI_w_spatial = as.factor(sp_data@meta.data$DGI_spatial_weights),
                          # DGI_expr = as.factor(sp_data@meta.data$DGI_expr_weights),
                          # DGI_w1 = as.factor(sp_data@meta.data$DGI_spatial_weights_1),
                          # VASC = as.factor(sp_data@meta.data$VASC),
                          # VASC_spatial = as.factor(sp_data@meta.data$VASC_with_spatial),
                          # GAE = as.factor(sp_data@meta.data$GAE),
                          # GAE_spatial = as.factor(sp_data@meta.data$GAE_with_spatial),
                          # VGAE = as.factor(sp_data@meta.data$VGAE),
                          # VGAE_spatial = as.factor(sp_data@meta.data$VGAE_with_spatial),
                          DGI = as.factor(sp_data@meta.data$DGI),
                          DGI_spatial = as.factor(sp_data@meta.data$DGI_with_spatial),
                          # APPNP = as.factor(sp_data@meta.data$APPNP),
                          # GAT = as.factor(sp_data@meta.data$GAT),
                          # RGCN = as.factor(sp_data@meta.data$RGCN),
                          SEDR = as.factor(sp_data@meta.data$SEDR),
                          Seurat = as.factor(sp_data@meta.data$Seurat),
                          SpaGCN = as.factor(sp_data@meta.data$SpaGCN),
                          BayesSpace = as.factor(sp_data@meta.data$BayesSpace),
                          Giotto = as.factor(sp_data@meta.data$Giotto),
                          stLearn = as.factor(sp_data@meta.data$stLearn))

df <- sp_data@images$slice1@coordinates
df <- cbind(df, df_clusters)

p0 <- ggplot(df, aes(imagecol, imagerow, color=layer_guess)) + geom_point(stroke=0, size=1.1) + ggtitle('layer_guess') +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

p1 <- ggplot(df, aes(imagecol, imagerow, color=Seurat)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('Seurat\nARI=', round(Seurat_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

p2 <- ggplot(df, aes(imagecol, imagerow, color=Giotto)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('Giotto\nARI=', round(Giotto_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

p3 <- ggplot(df, aes(imagecol, imagerow, color=stLearn)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('stLearn\nARI=', round(stLearn_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

p4 <- ggplot(df, aes(imagecol, imagerow, color=SpaGCN)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('SpaGCN\nARI=', round(SpaGCN_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

p5 <- ggplot(df, aes(imagecol, imagerow, color=BayesSpace)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('BayesSpace\nARI=', round(BayesSpace_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

p6 <- ggplot(df, aes(imagecol, imagerow, color=SEDR)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('SEDR\nARI=', round(SEDR_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

# p7 <- ggplot(df, aes(imagecol, imagerow, color=APPNP)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('APPNP\nARI=', round(APPNP_ARI, 3))) +
#   coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

# p8 <- ggplot(df, aes(imagecol, imagerow, color=GAT)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('GAT\nARI=', round(GAT_ARI, 3))) +
# coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

# p9 <- ggplot(df, aes(imagecol, imagerow, color=RGCN)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('RGCN\nARI=', round(RGCN_ARI, 3))) +
# coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

# p10 <- ggplot(df, aes(imagecol, imagerow, color=DGI_w1)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('GCN\n(edge weight = 1)\nARI=', round(DGI_w1_ARI, 3))) +
# coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

# p11 <- ggplot(df, aes(imagecol, imagerow, color=DGI_expr)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('GCN\n(edge wt = expr.sim)\nARI=', round(DGI_expr_ARI, 3))) +
# coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

# p12 <- ggplot(df, aes(imagecol, imagerow, color=DGI_spatial)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('GCN\n(edge wt = 1 - sp.dist)\nARI=', round(DGI_spatial_ARI, 3))) +
# coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

# p13 <- ggplot(df, aes(imagecol, imagerow, color=DGI_expr_spatial)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('GCN\n(edge wt=(1-sp.dist)*expr.sim)\nARI=', round(DGI_expr_spatial_ARI, 3))) +
# coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

# p14 <- ggplot(df, aes(imagecol, imagerow, color=VASC)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('VASC\nARI=', round(VASC_ARI, 3))) +
# coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

# p15 <- ggplot(df, aes(imagecol, imagerow, color=VASC_spatial)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('VASC+SP\nARI=', round(VASC_spatial_ARI, 3))) +
# coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

# p16 <- ggplot(df, aes(imagecol, imagerow, color=GAE)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('GAE\nARI=', round(GAE_ARI, 3))) +
# coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

# p17 <- ggplot(df, aes(imagecol, imagerow, color=GAE_spatial)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('GAE+SP\nARI=', round(GAE_spatial_ARI, 3))) +
# coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

# p18 <- ggplot(df, aes(imagecol, imagerow, color=VGAE)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('VGAE\nARI=', round(VGAE_ARI, 3))) +
# coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

# p19 <- ggplot(df, aes(imagecol, imagerow, color=VGAE_spatial)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('VGAE+SP\nARI=', round(VGAE_spatial_ARI, 3))) +
# coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

p20 <- ggplot(df, aes(imagecol, imagerow, color=DGI)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('DGI\nARI=', round(DGI_ARI, 3))) +
coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

p21 <- ggplot(df, aes(imagecol, imagerow, color=DGI_spatial)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('DGI+SP\nARI=', round(DGI_spatial_ARI, 3))) +
coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T) + theme(plot.title = element_text(hjust = 0.5))

p0 + p1 + p2 + p3 + p4 + p5 + p20 + p21 + plot_layout(ncol = 4, nrow = 2, widths = c(1,1,1,1), heights = c(1,1)) & NoLegend()#+ p7 + p8 + p9 + p10 + p11 + p12 + p13 + p14+ p15 + p16 + p17 + p18 + p19 



dir.output <- file.path('../output/DLPFC/', sample, '/Comparison/')
if(!dir.exists(file.path(dir.output))){
  dir.create(file.path(dir.output), recursive = TRUE)
}


ggsave(filename = file.path(dir.output, 'comparison.png'), width=9, height=5)
ggsave(filename = file.path(dir.output,  'comparison.pdf'), width=9, height=5)

write.table(df, file.path(dir.output, 'comparison.tsv'), sep='\t', quote=FALSE)

