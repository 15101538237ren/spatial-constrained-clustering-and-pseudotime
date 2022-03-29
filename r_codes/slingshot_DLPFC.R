library(SingleCellExperiment)
library(slingshot)
library(loomR)
samples_list = c("151673")#"151510", "151507", 
for (sample in samples_list){
  input_dir = paste0("../data/DLPFC/",sample,"/export");
  pcs= read.table(paste(c(input_dir, "pcs.tsv"), collapse = '/'), header=T, sep='\t')
  leiden_clusters = read.table(paste(c(input_dir, "leiden.tsv"), collapse = '/'), header=T, sep='\t')
  
  loom_fp <- paste(c(input_dir, "adata_filtered_cp.loom"), collapse = "/")
  loom_obj <- connect(loom_fp,mode = 'r+',skip.validate = T)
  counts = as(t(loom_obj[["matrix"]][,]), "sparseMatrix")
  nPCs = 5
  
  sce <- SingleCellExperiment(assays = list(counts = counts), 
                              reducedDims = SimpleList(PCA = as.matrix(pcs[,c(1:nPCs)])),
                              colData = data.frame(leiden = leiden_clusters$leiden_label))
  
  sce <- slingshot(sce, clusterLabels = 'leiden', reducedDim = 'PCA')
  
  dir.output <-paste0("../output/DLPFC/",sample,"/slingshot");
  dir.create(dir.output, showWarnings = F)
  
  write.table(sce$slingPseudotime_1, file = file.path(dir.output, 'pseudotime.tsv'), sep='\t', quote=F, row.names = F, col.names = F)
}
