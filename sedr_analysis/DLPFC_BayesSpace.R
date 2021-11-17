library(BayesSpace)
library(ggplot2)
args = commandArgs(trailingOnly=TRUE)
sample.names <- c('151507', '151508', '151509', '151510', '151669', '151670', '151672', '151673', '151674', '151675', '151676', '151671')
n_clusters <- c(7, 7, 7, 7, 5, 5, 5, 7, 7, 7, 7, 5)
for (i in 1:length(n_clusters)){
  sample.name = sample.names[i]
  n_cluster = n_clusters[i]
  print(sprintf("Running on %s", sample.name))
  dir.input <- file.path('../data/DLPFC/', sample.name)
  dir.output <- file.path('../output/DLPFC/', sample.name, '/BayesSpace/')
  
  if(!dir.exists(file.path(dir.output))){
    dir.create(file.path(dir.output), recursive = TRUE)
  }
  
  
  dlpfc <- getRDS("2020_maynard_prefrontal-cortex", sample.name)
  
  set.seed(101)
  dec <- scran::modelGeneVar(dlpfc)
  top <- scran::getTopHVGs(dec, n = 2000)
  
  set.seed(102)
  dlpfc <- scater::runPCA(dlpfc, subset_row=top)
  
  ## Add BayesSpace metadata
  dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=TRUE)
  
  
  ##### Clustering with BayesSpace
  q <- n_cluster  # Number of clusters
  d <- 15  # Number of PCs
  
  ## Run BayesSpace clustering
  set.seed(104)
  dlpfc <- spatialCluster(dlpfc, q=q, d=d, platform='Visium',
                          nrep=50000, gamma=3, save.chain=TRUE)
  
  labels <- dlpfc$spatial.cluster
  
  ## View results
  clusterPlot(dlpfc, label=labels, palette=NULL, size=0.05) +
    scale_fill_viridis_d(option = "A", labels = 1:7) +
    labs(title="BayesSpace")
  
  ggsave(file.path(dir.output, 'clusterPlot.png'), width=5, height=5)
  
  
  write.table(colData(dlpfc), file=file.path(dir.output, 'metadata.tsv'), sep='\t', quote=FALSE)
  
}
