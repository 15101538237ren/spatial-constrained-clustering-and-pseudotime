library(biomaRt)
ensembl = useMart("ensembl",dataset="hsapiens_gene_ensembl") #uses human ensembl annotations
gene.data <- getBM(attributes=c('hgnc_symbol', 'go_id'), mart = ensembl)
gene.data = gene.data[gene.data$go_id!="",]
write.csv(gene.data,"../data/Visium/Breast_Cancer/analysis/genes_with_go_ids.csv", row.names = F, quote = T)


gene.data <- getBM(attributes=c('go_id', 'name_1006', 'definition_1006'), mart = ensembl)
gene.data = gene.data[gene.data$go_id!="",]
write.csv(gene.data,"../data/Visium/Breast_Cancer/analysis/genes_with_go_ids_and_def.csv", row.names = F, quote = T)