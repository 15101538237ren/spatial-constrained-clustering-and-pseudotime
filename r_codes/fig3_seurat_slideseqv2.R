library(Seurat)
library(SeuratData)
InstallData("ssHippo")
slide.seq <- LoadData("ssHippo")
slide.seq <- SCTransform(slide.seq, assay = "Spatial", ncells = 3000, verbose = FALSE)
slide.seq <- RunPCA(slide.seq)
slide.seq <- RunUMAP(slide.seq, dims = 1:50)
slide.seq <- FindNeighbors(slide.seq, dims = 1:50)
slide.seq <- FindClusters(slide.seq, resolution = 0.5, verbose = FALSE)


dir.output <- "../output/slideseq_v2/slideseq_v2/DGI_SP/Seurat"
dir.create(dir.output, showWarnings = F)

write.table(slide.seq@reductions$pca@cell.embeddings, file = file.path(dir.output, 'seurat.PCs.tsv'), sep='\t', quote=F)

write.table(slide.seq@meta.data, file = file.path(dir.output, './metadata.tsv'), sep='\t', quote=FALSE)