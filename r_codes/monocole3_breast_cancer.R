library(monocle)
library(loomR)

input_dir = "../data/Visium/breast_cancer/export/G1"
loom_fp <- paste(c(input_dir, "adata.loom"), collapse = "/")
loom_obj <- connect(loom_fp,mode = 'r+',skip.validate = T)

expr_matrix=t(loom_obj[["matrix"]][,])
gene=loom_obj$row.attrs$var_names[]
barcode=loom_obj$col.attrs$obs_names[]

pd <- new("AnnotatedDataFrame", data = data.frame(barcode))
fd <- new("AnnotatedDataFrame", data = data.frame(gene))

breast_cancer_data <- newCellDataSet(as(expr_matrix, "sparseMatrix"), phenoData = pd, featureData = fd)

breast_cancer_data <- estimateSizeFactors(breast_cancer_data)
breast_cancer_data <- estimateDispersions(breast_cancer_data)

breast_cancer_data <- reduceDimension(breast_cancer_data, max_components = 2, method = 'DDRTree')
breast_cancer_data <- orderCells(breast_cancer_data)
plot_cell_trajectory(breast_cancer_data)
plot_cell_trajectory(breast_cancer_data, color_by = "Pseudotime")



dir.output <- "../output/breast_cancer/G1/monocole"
dir.create(dir.output, showWarnings = F)

write.table(breast_cancer_data$Pseudotime, file = file.path(dir.output, 'pseudotime.tsv'), sep='\t', quote=F, row.names = F, col.names = F)









