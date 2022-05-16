library(BayesSpace)
library(ggplot2)
library(Matrix)

input_dir = "../data/stereo_seq/"

counts <- read.table(paste(c(input_dir, "RNA_counts.tsv.gz"), collapse = "/"),
                   sep="\t",header=TRUE)
rownames(counts)<-counts$geneID
counts <- counts[,-c(1)]
colnames(counts)<-sprintf("%s",seq(1:ncol(counts)))
count_val<-as(as.matrix(counts[1:nrow(counts),1:ncol(counts)]), "dgCMatrix")
sce <- SingleCellExperiment(assays=list(counts=count_val),
                            rowData=rownames(counts),
                            colData=colnames(counts))
sample.name <- "stereo_seq"
n_clusters <- 8

print(sprintf("Running on %s", sample.name))

dir.output <- "../output/stereo_seq/stereo_seq/BayesSpace"
if(!dir.exists(file.path(dir.output))){
  dir.create(file.path(dir.output), recursive = TRUE)
}

set.seed(101)
sce <- spatialPreprocess(sce, platform="ST", 
                              n.PCs=15, n.HVGs=2000, log.normalize=T)

dec <- scran::modelGeneVar(sce)
top <- scran::getTopHVGs(dec, n = 2000)

set.seed(102)
sce <- scater::runPCA(sce, subset_row=top)

## Add BayesSpace metadata
sce <- spatialPreprocess(sce, platform="stereo-seq", skip.PCA=TRUE)

set.seed(149)
sce <- spatialCluster(sce, q=8, platform="ST", d=15,
                           init.method="mclust", model="t", gamma=2,
                           nrep=1000, burn.in=100,
                           save.chain=T)

##### Clustering with BayesSpace
q <- n_clusters  # Number of clusters
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
