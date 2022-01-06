library(Seurat)
library(loomR)

input_dir = "../data/seqfish_mouse/seqfish_mouse/export"
loom_fp <- paste(c(input_dir, "adata.loom"), collapse = "/")
loom_obj <- connect(loom_fp,mode = 'r+',skip.validate = T)

seqfish=t(loom_obj[["matrix"]][,])
gene=loom_obj$row.attrs$var_names[]
barcode=loom_obj$col.attrs$obs_names[]
colnames(seqfish)= barcode
row.names(seqfish)= gene
seqfish=CreateSeuratObject(counts = seqfish,project = 'seqfish', assay = "Spatial",min.cells = 3, min.features = 0)

seqfish <- SCTransform(seqfish, assay = "Spatial", verbose = F)
seqfish <- RunPCA(seqfish, assay = "SCT", verbose = F)
seqfish <- RunUMAP(seqfish, reduction = "pca", dims = 1:50)
seqfish <- FindNeighbors(seqfish, reduction = "pca", dims = 1:50)
seqfish <- FindClusters(seqfish, resolution = 1.0, verbose = F)

dir.output <- "../output/seqfish_mouse/seqfish_mouse/Seurat"
dir.create(dir.output, showWarnings = F)

write.table(seqfish@reductions$pca@cell.embeddings, file = file.path(dir.output, 'seurat.PCs.tsv'), sep='\t', quote=F)

write.table(seqfish@meta.data, file = file.path(dir.output, './metadata.tsv'), sep='\t', quote=F)