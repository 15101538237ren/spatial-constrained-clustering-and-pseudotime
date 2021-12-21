if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

#BiocManager::install(c("topGO", "GO.db", "biomaRt", "Rgraphviz"))

library(topGO)
library(GO.db)
library(biomaRt)
library(Rgraphviz)

output_dir = "../output"
dataset_name = "breast_cancer"
sample_name = "G1"
method_name = "DGI_SP"
subdir_name = "cluster_specific_marker_genes"

input_dir_c = c(output_dir, dataset_name, sample_name, method_name, subdir_name)
input_dir_fp = paste(input_dir_c, collapse = '/')
gene_back_groud_filename = "background_genes.tsv"
clusters = c('APC,B,T-1', 'APC,B,T-2', 'Connective', 'Imm&Conn', 'Imm-Reg-1', 'Imm-Reg-2'
             , 'Interface-1', 'Interface-2', 'Interface-3', 'Invasive')

# Step 1: Preparing the data in the required format
# Gene universe file
exp_data= read.table(paste(c(input_dir_fp, gene_back_groud_filename), collapse = '/'), header=F, sep='\t')
bg_genes=as.character(exp_data[, 1])

# Step 2: Create GO annotation
# create GO db for genes to be used using biomaRt - please note that this takes a while
# To see the different datasets available within a biomaRt you can e.g. do: mart = useMart('ensembl'), followed by listDatasets(mart)
db= useMart('ENSEMBL_MART_ENSEMBL',dataset='hsapiens_gene_ensembl')
go_ids= getBM(attributes=c('go_id', 'external_gene_name', 'namespace_1003'), filters='external_gene_name', values=bg_genes, mart=db)

# build the gene 2 GO annotation list (needed to create topGO object)
gene_2_GO=unstack(go_ids[,c(1,2)])

for (cluster in clusters)
{
  marker_genes_filename = paste0(cluster, ".tsv")
  
  # Read in genes of interest
  candidate_df =read.table(paste(c(input_dir_fp, marker_genes_filename), collapse = '/'), header=F, sep='\t')
  candidate_list= as.character(candidate_df[,1])
  
  # remove any candidate genes without GO annotation
  keep = candidate_list %in% go_ids[,2]
  candidate_list=candidate_list[which(keep==TRUE)]
  
  # make named factor showing which genes are of interest
  geneList=factor(as.integer(bg_genes %in% candidate_list))
  names(geneList)= bg_genes
  
  # Step 3: Make topGO data object
  GOdata=new('topGOdata', ontology='BP', allGenes = geneList, annot = annFUN.gene2GO, gene2GO = gene_2_GO)
  
  # Step 4: Test for significance
  # define test using the classic algorithm with fisher (refer to [1] if you want to understand how the different algorithms work)
  classic_fisher_result=runTest(GOdata, algorithm='classic', statistic='fisher')
  
  # define test using the weight01 algorithm (default) with fisher
  weight_fisher_result=runTest(GOdata, algorithm='weight01', statistic='fisher') 
  
  # generate a table of results: we can use the GenTable function to generate a summary table with the results from tests applied to the topGOdata object.
  allGO=usedGO(GOdata)
  all_res=GenTable(GOdata, weightFisher=weight_fisher_result, orderBy='weightFisher', topNodes=length(allGO), numChar=1000)
  
  
  # Step 5: Correcting for multiple testing
  #performing BH correction on our p values
  p.adj=round(p.adjust(all_res$weightFisher,method="BH"),digits = 4)
  
  # create the file with all the statistics from GO analysis
  all_res_final=cbind(all_res,p.adj)
  all_res_final=all_res_final[order(all_res_final$p.adj),]
  
  #get list of significant GO before multiple testing correction
  results.table.p= all_res_final[which(all_res_final$weightFisher<0.005),]
  
  #save first top 50 ontolgies sorted by adjusted pvalues
  target_file_name = paste(c("topGO_terms.tsv"), collapse = '')
  
  results.table.p$log10WF=-log10(as.numeric(results.table.p$weightFisher))
  
  # go_term_fp = paste(c(input_dir_fp, target_file_name), collapse = '/')
  # write.table(results.table.p, go_term_fp,sep="\t",quote=FALSE,row.names=FALSE)
  
  
  library(ggplot2)
  library(dplyr)
  
  # results.table.p =read.table(go_term_fp, header=T, sep='\t')
  
  # Reorder following the value of another column:
  results.table.p %>%
    mutate(Term = reorder(Term, log10WF)) %>%
    ggplot( aes(x=Term, y=log10WF,fill = log10WF)) +
    geom_bar(stat="identity", alpha=.6, width=.4) +
    coord_flip() +
    scale_fill_gradient(name="-log10(adj.pval)", low="blue", high="red",
                        limits = c(2.0, 4.0), 
                        breaks = c(2.1, 3, 3.9),
                        labels = c(2.0, 3, 4.0)) +
    xlab("") + ylab("-log10(adj.pval)") + theme_bw() + 
    theme(axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank(),
          axis.text.x = element_text(colour = "black", size=8, face="bold"),
          axis.text.y = element_text(colour = "black", size=8, face="bold"),
          axis.title.x = element_text(colour = "black", size=8, face="bold"),
          legend.title = element_text(size=8, face="bold"),
          legend.text = element_text(size=8, face="bold"),
          legend.key.height= unit(.3, 'cm'),
          legend.key.width= unit(.3, 'cm'))
  
  target_fig_name = paste0(cluster, "_topGO.svg")
  ggsave(paste(c(input_dir_fp, target_fig_name), collapse = '/'), dpi=300) #, width=5.5, height=4
}

