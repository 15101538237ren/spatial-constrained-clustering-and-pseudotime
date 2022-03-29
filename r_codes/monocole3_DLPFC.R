library(monocle)
library(loomR)

samples_list = c('151510', '151507', '151673')

for (sample in samples_list){
  input_dir = paste0("../data/DLPFC/",sample,"/export");
  loom_fp <- paste(c(input_dir, "adata_filtered.loom"), collapse = "/")
  loom_obj <- connect(loom_fp,mode = 'r+',skip.validate = T)
  
  expr_matrix=t(loom_obj[["matrix"]][,])
  gene=loom_obj$row.attrs$var_names[]
  barcode=loom_obj$col.attrs$obs_names[]
  
  pd <- new("AnnotatedDataFrame", data = data.frame(barcode))
  fd <- new("AnnotatedDataFrame", data = data.frame(gene))
  
  dlpfc <- newCellDataSet(as(expr_matrix, "sparseMatrix"), phenoData = pd, featureData = fd)
  
  dlpfc <- estimateSizeFactors(dlpfc)
  dlpfc <- estimateDispersions(dlpfc)
  
  dlpfc <- reduceDimension(dlpfc, max_components = 2, method = 'DDRTree')
  dlpfc <- orderCells(dlpfc)
  plot_cell_trajectory(dlpfc)
  plot_cell_trajectory(dlpfc, color_by = "Pseudotime")
  
  
  dir.output <-paste0("../output/DLPFC/",sample,"/monocle");
  
  dir.create(dir.output, showWarnings = F)
  
  write.table(dlpfc$Pseudotime, file = file.path(dir.output, 'pseudotime.tsv'), sep='\t', quote=F, row.names = F, col.names = F)
}









