library(Seurat)
library(SeuratDisk)

input_dir = "../data/seqfish_mouse/seqfish_mouse/export"
h5ad_fp = paste(c(input_dir, "adata.h5ad"), collapse = "/")
h5seurat_fp = paste(c(input_dir, "adata.h5seurat"), collapse = "/")

Convert(h5ad_fp, dest = "h5seurat", assay = "Spatial", overwrite = T)
seqfish <- LoadH5Seurat(h5seurat_fp)
seqfish <- SCTransform(seqfish, assay = "Spatial", verbose = F)
seqfish <- RunPCA(seqfish, assay = "SCT", verbose = F)
seqfish <- RunUMAP(seqfish, reduction = "pca", dims = 1:50)
seqfish <- FindNeighbors(seqfish, reduction = "pca", dims = 1:50)
seqfish <- FindClusters(seqfish, resolution = 0.5, verbose = F)

dir.output <- "../output/seqfish_mouse/seqfish_mouse/DGI_SP/Seurat"
dir.create(dir.output, showWarnings = F)

write.table(seqfish@reductions$pca@cell.embeddings, file = file.path(dir.output, 'seurat.PCs.tsv'), sep='\t', quote=F)

write.table(seqfish@meta.data, file = file.path(dir.output, './metadata.tsv'), sep='\t', quote=F)