args = commandArgs(trailingOnly=TRUE)
sample.name <- args[1]
n_clusters <- args[2]


library(Seurat)
library(SeuratData)
library(ggplot2)
library(patchwork)
library(dplyr)
options(bitmapType = 'cairo')


dir.input <- file.path('../data/DLPFC/', sample.name)
dir.output <- file.path('../output/DLPFC/', sample.name, '/Seurat/')

if(!dir.exists(file.path(dir.output))){
  dir.create(file.path(dir.output), recursive = TRUE)
}

### load data
sp_data <- Load10X_Spatial(dir.input, filename = "filtered_feature_bc_matrix.h5")

df_meta <- read.table(file.path(dir.input, 'metadata.tsv'))

sp_data <- AddMetaData(sp_data, 
                       metadata = df_meta$layer_guess,
                       col.name = 'layer_guess')

### Data processing
plot1 <- VlnPlot(sp_data, features = "nCount_Spatial", pt.size = 0.1) + NoLegend()
plot2 <- SpatialFeaturePlot(sp_data, features = "nCount_Spatial") + theme(legend.position = "right")
wrap_plots(plot1, plot2)
ggsave(file.path(dir.output, './Seurat.QC.png'), width = 10, height=5)

# sctransform
sp_data <- SCTransform(sp_data, assay = "Spatial", verbose = FALSE)


### Dimensionality reduction, clustering, and visualization
sp_data <- RunPCA(sp_data, assay = "SCT", verbose = FALSE, npcs = 50)
sp_data <- FindNeighbors(sp_data, reduction = "pca", dims = 1:30)

for(resolution in 50:30){
  sp_data <- FindClusters(sp_data, verbose = F, resolution = resolution/100)
  if(length(levels(sp_data@meta.data$seurat_clusters)) == n_clusters){
    break
  }
}
sp_data <- FindClusters(sp_data, verbose = FALSE, resolution = 0.46)
sp_data <- RunUMAP(sp_data, reduction = "pca", dims = 1:30)

p1 <- DimPlot(sp_data, reduction = "umap", label = TRUE)
p2 <- SpatialDimPlot(sp_data, label = TRUE, label.size = 3)
p1 + p2
ggsave(file.path(dir.output, './Seurat.cell_cluster.png'), width=10, height=5)


##### save data
saveRDS(sp_data, file.path(dir.output, 'Seurat_final.rds'))

write.table(sp_data@reductions$pca@cell.embeddings, file = file.path(dir.output, 'seurat.PCs.tsv'), sep='\t', quote=F)

write.table(sp_data@meta.data, file = file.path(dir.output, './metadata.tsv'), sep='\t', quote=FALSE)


##### 
library(mclust)

print(adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$seurat_clusters))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           le to temp file /tmp/RtmpoB5LdP/file22496fc04d93
17:37:22 Searching Annoy index using 1 thread, search_k = 3000
17:37:26 Annoy recall = 100%
17:37:27 Commencing smooth kNN distance calibration using 1 thread
17:37:28 Initializing from normalized Laplacian + noise
17:37:28 Commencing optimization for 500 epochs, with 195628 positive edges
0%   10   20   30   40   50   60   70   80   90   100%
[----|----|----|----|----|----|----|----|----|----|
**************************************************|
17:37:39 Optimization finished
[?25h[?25hScale for 'fill' is already present. Adding another scale for 'fill', which
will replace the existing scale.
[?25h[?25h[?25h[?25h[?25h[?25hPackage 'mclust' version 5.4.7
Type 'citation("mclust")' for citing this R package in publications.
[?25h[1] 0.3790042
[?25hError: invalid multibyte character in parser at line 1
Execution halted
[?25hError: unexpected symbol in "Script started"
Execution halted
Error: unexpected symbol in "Script started"
Execution halted
Error: unexpected symbol in "Script started"
Execution halted
Error: unexpected symbol in "Script started"
Execution halted
Error: unexpected symbol in "Script started"
Execution halted
Error: unexpected symbol in "Script started"
Execution halted
Error: unexpected symbol in "Script started"
Execution halted
Error: unexpected symbol in "Script started"
Execution halted
Error: unexpected symbol in "Script started"
Execution halted
Error: unexpected symbol in "Script started"
Execution halted
]0;~/Xuhang/Projects/spTrans_1/script
[32mxuhang@scHulk [36m[17:38:37][0m [33m~/Xuhang/Projects/spTrans_1/script[0m
$ ls ..//[Koutput/DLPFC/151[K[K[K*
../output/DLPFC/151507:
[0m[01;34mBayesSpace[0m  [01;34mSEDR[0m  [01;34mSeurat[0m

../output/DLPFC/151508:
[01;34mSEDR[0m  [01;34mSeurat[0m

../output/DLPFC/151509:
[01;34mSEDR[0m

../output/DLPFC/151510:
[01;34mSEDR[0m

../output/DLPFC/151669:
[01;34mSEDR[0m

../output/DLPFC/151670:
[01;34mSEDR[0m

../output/DLPFC/151671:
[01;34mSEDR[0m

../output/DLPFC/151672:
[01;34mSEDR[0m

../output/DLPFC/151673:
[01;34mSEDR[0m
]0;~/Xuhang/Projects/spTrans_1/script
[32mxuhang@scHulk [36m[17:38:54][0m [33m~/Xuhang/Projects/spTrans_1/script[0m
$ ls ../output/DLPFC/*[K[K[KC/151508
[0m[01;34mSEDR[0m  [01;34mSeurat[0m
]0;~/Xuhang/Projects/spTrans_1/script
[32mxuhang@scHulk [36m[17:39:00][0m [33m~/Xuhang/Projects/spTrans_1/script[0m
$ ls ../output/DLPFC/151508Seu
ls: cannot access '../output/DLPFC/151508Seu': No such file or directory
]0;~/Xuhang/Projects/spTrans_1/script
[32mxuhang@scHulk [36m[17:39:02][0m [33m~/Xuhang/Projects/spTrans_1/script[0m
$ ls ../output/DLPFC/151508Seu[K[K[K/Su[Keurat/
metadata.tsv  [0m[01;35mSeurat.cell_cluster.png[0m  Seurat_final.rds  seurat.PCs.tsv  [01;35mSeurat.QC.png[0m
]0;~/Xuhang/Projects/spTrans_1/script
[32mxuhang@scHulk [36m[17:39:05][0m [33m~/Xuhang/Projects/spTrans_1/script[0m
$ ls ../output/DLPFC/151508/Seurat/[5PSeu[K*[K[C[CRscript DLPFC_Seurat.R 151676 75[C[C6[C[C5[C[C4[C[C3[C[C2 51[C[C0[C[C69[C[C510 709[C[C
Error: unexpected symbol in "Script started"
Execution halted
]0;~/Xuhang/Projects/spTrans_1/script
[32mxuhang@scHulk [36m[17:39:15][0m [33m~/Xuhang/Projects/spTrans_1/script[0m
$ Rscript DLPFC_Seurat.R 151509 7
Rscript DLPFC_Seurat.R 151510 7
Attaching SeuratObject
Registered S3 method overwritten by 'cli':
  method     from         
  print.boxx spatstat.geom
── [1mInstalled datasets[22m ───────────────────────────────────────────────────────────────────────────────────────────────────────────────── SeuratData v0.2.1 ──
[32m✔[39m [34mbmcite  [39m 0.3.0                                                              [32m✔[39m [34mpbmc3k  [39m 3.1.4
[32m✔[39m [34mifnb    [39m 3.0.0                                                              [32m✔[39m [34mssHippo [39m 3.1.4
[32m✔[39m [34mpanc8   [39m 3.0.2                                                              [32m✔[39m [34mstxBrain[39m 0.1.1

──────────────────────────────────────────────────────────────────────────── Key ───────────────────────────────────────────────────────────────────────────
[32m✔[39m Dataset loaded successfully
[33m❯[39m Dataset built with a newer version of Seurat than installed
[31m❓[39m Unknown version of Seurat installed

There were 12 warnings (use warnings() to see them)
[?25h[?25h[?25h
Attaching package: ‘dplyr’

The following objects are masked from ‘package:stats’:

    filter, lag

The following objects are masked from ‘package:base’:

    intersect, setdiff, setequal, union

[?25h[?25h[?25h[?25h[?25hWarning message:
In sparseMatrix(i = indices[] + 1, p = indptr[], x = as.numeric(x = counts[]),  :
  'giveCsparse' has been deprecated; setting 'repr = "T"' for you
[?25h[?25h[?25h[?25hWarning message:
`guides(<scale> = FALSE)` is deprecated. Please use `guides(<scale> = "none")` instead. 
[?25h[?25h[?25hThere were 50 or more warnings (use warnings() to see the first 50)
[?25h[?25hComputing nearest neighbor graph
Computing SNN
[?25h[?25h[?25hWarning: The default method for RunUMAP has changed from calling Python UMAP via reticulate to the R-native UWOT using the cosine metric
To use Python UMAP via reticulate, set umap.method to 'umap-learn' and metric to 'correlation'
This message will be shown once per session
17:53:57 UMAP embedding parameters a = 0.9922 b = 1.112
17:53:57 Read 4789 rows and found 30 numeric columns
17:53:57 Using Annoy for neighbor search, n_neighbors = 30
17:53:57 Building Annoy index with metric = cosine, n_trees = 50
0%   10   20   30   40   50   60   70   80   90   100%
[----|----|----|----|----|----|----|----|----|----|
**************************************************|
17:53:59 Writing NN index file to temp file /tmp/RtmpCBISCA/file50213960aa12
17:53:59 Searching Annoy index using 1 thread, search_k = 3000
17:54:02 Annoy recall = 100%
17:54:02 Commencing smooth kNN distance calibration using 1 thread
17:54:04 Initializing from normalized Laplacian + noise
17:54:04 Commencing optimization for 500 epochs, with 215164 positive edges
0%   10   20   30   40   50   60   70   80   90   100%
[----|----|----|----|----|----|----|----|----|----|
**************************************************|
17:54:16 Optimization finished
[?25h[?25hScale for 'fill' is already present. Adding another scale for 'fill', which
will replace the existing scale.
[?25h[?25h[?25h[?25h[?25h[?25hPackage 'mclust' version 5.4.7
Type 'citation("mclust")' for citing this R package in publications.
[?25h[1] 0.25272
[?25h[?25hAttaching SeuratObject
Registered S3 method overwritten by 'cli':
  method     from         
  print.boxx spatstat.geom
── [1mInstalled datasets[22m ───────────────────────────────────────────────────────────────────────────────────────────────────────────────── SeuratData v0.2.1 ──
[32m✔[39m [34mbmcite  [39m 0.3.0                                                              [32m✔[39m [34mpbmc3k  [39m 3.1.4
[32m✔[39m [34mifnb    [39m 3.0.0                                           