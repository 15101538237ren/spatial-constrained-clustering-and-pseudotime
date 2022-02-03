library(Seurat)
library(loomR)
input_dir = "../data/slideseq_v2/slideseq_v2/export"
loom_fp <- paste(c(input_dir, "adata.loom"), collapse = "/")
loom_obj <- connect(loom_fp,mode = 'r+',skip.validate = T)

slide.seq=t(loom_obj[["matrix"]][,])
gene=loom_obj$row.attrs$var_names[]
barcode=loom_obj$col.attrs$obs_names[]
colnames(slide.seq)= barcode
row.names(slide.seq)=  gene
slide.seq=CreateSeuratObject(counts = slide.seq,project = 'slide.seq', assay = "Spatial",min.cells = 3, min.features = 0)

slide.seq <- SCTransform(slide.seq, assay = "Spatial", verbose = F)
slide.seq <- RunPCA(slide.seq)
slide.seq <- RunUMAP(slide.seq, dims = 1:50)
slide.seq <- FindNeighbors(slide.seq, dims = 1:50)
slide.seq <- FindClusters(slide.seq, resolution = 0.4, verbose = F)


dir.output <- "../output/slideseq_v2/slideseq_v2/Seurat"
dir.create(dir.output, showWarnings = F)

write.table(slide.seq@reductions$pca@cell.embeddings, file = file.path(dir.output, 'seurat.PCs.tsv'), sep='\t', quote=F, row.names = F)

write.table(slide.seq@meta.data, file = file.path(dir.output, './metadata.tsv'), sep='\t', quote=F)