require(biomaRt)

filter_non_human_genes <- function(data.input){
  genes_with_ens <- rownames(py_to_r(ad_object$var))
  
  # chicken <- useMart('ensembl', dataset = 'ggallus_gene_ensembl')
  # human <- useMart('ensembl', dataset = 'hsapiens_gene_ensembl')
  # 
  # indices_with_ens <- startsWith(genes_with_ens, "ENSGALG")
  # 
  # ensnames <- genes_with_ens[indices_with_ens]
  # 
  # master_ens <- getLDS(
  #   filters = 'ensembl_gene_id',
  #   values = ensnames, 
  #   mart = chicken, 
  #   attributes = c('ensembl_gene_id','external_gene_name','entrezgene_id'),
  #   martL = human, 
  #   attributesL = c('hgnc_symbol','ensembl_gene_id','entrezgene_id','gene_biotype'))
  # 
  # master_filtered <- master_ens[!(master_ens$HGNC.symbol==""), ]
  # 
  # master_filtered <- master_filtered[!duplicated(master_filtered$Gene.stable.ID), ]
  # indices_to_replace <- match(master_filtered$Gene.stable.ID, genes_with_ens)
  # genes_with_ens[indices_to_replace] <- master_filtered$HGNC.symbol
  
  indices_with_ens <- startsWith(genes_with_ens, "ENSGALG")
  
  ensnames <- genes_with_ens[!indices_with_ens]
  data.input <- data.input[!indices_with_ens, ]
  
  rownames(data.input) <- ensnames
  colnames(data.input) <- rownames(py_to_r(ad_object$obs))
  
  return(data.input)
}