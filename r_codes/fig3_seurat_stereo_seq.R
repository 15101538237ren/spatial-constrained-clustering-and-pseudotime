library(Seurat)
library(SeuratDisk)

input_dir = "../data/stereo_seq/stereo_seq/export"
h5ad_fp = paste(c(input_dir, "adata.h5ad"), collapse = "/")
h5seurat_fp = paste(c(input_dir, "adata.h5seurat"), collapse = "/")

Convert(h5ad_fp, dest = "h5seurat", assay = "Spatial", overwrite = T)
stereo.seq <- LoadH5Seurat(h5seurat_fp)
stereo.seq <- SCTransform(stereo.seq, assay = "Spatial", verbose = F)
stereo.seq <- RunPCA(stereo.seq, assay = "SCT", verbose = F)
stereo.seq <- RunUMAP(stereo.seq, dims = 1:50)
stereo.seq <- FindNeighbors(stereo.seq, dims = 1:50)
stereo.seq <- FindClusters(stereo.seq, resolution = 0.4, verbose = F)

dir.output <- "../output/stereo_seq/stereo_seq/Seurat"
dir.create(dir.output, showWarnings = F)

write.table(stereo.seq@reductions$pca@cell.embeddings, file = file.path(dir.output, 'seurat.PCs_0.4.tsv'), sep='\t', quote=F, row.names = F)

write.table(stereo.seq@meta.data, file = file.path(dir.output, './metadata_0.4.tsv'), sep='\t', quote=F)


