library(SingleCellExperiment)
library(slingshot)
library(loomR)
input_dir = "../data/stereo_seq/stereo_seq/export"
pcs= read.table(paste(c(input_dir, "pcs.tsv"), collapse = '/'), header=T, sep='\t')
leiden_clusters = read.table(paste(c(input_dir, "leiden.tsv"), collapse = '/'), header=T, sep='\t')

loom_fp <- paste(c(input_dir, "adata.loom"), collapse = "/")
loom_obj <- connect(loom_fp,mode = 'r+',skip.validate = T)
counts = as(t(loom_obj[["matrix"]][,]), "sparseMatrix")
nPCs = 5

sce <- SingleCellExperiment(assays = list(counts = counts), 
                            reducedDims = SimpleList(PCA = as.matrix(pcs[,c(1:nPCs)])),
                            colData = data.frame(leiden = leiden_clusters$leiden_label))

sce <- slingshot(sce, clusterLabels = 'leiden', reducedDim = 'PCA')

dir.output <- "../output/stereo_seq/stereo_seq/slingshot"
dir.create(dir.output, showWarnings = F)

write.table(sce$slingPseudotime_1, file = file.path(dir.output, 'pseudotime.tsv'), sep='\t', quote=F, row.names = F)


