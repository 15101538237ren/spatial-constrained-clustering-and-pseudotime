library(MERINGUE)

seq_name<-"seqfish_mouse" #"DLPFC"
list.samples<-c("seqfish_mouse")#c("151508", "151509", "151510", "151669", "151670", "151672", "151673", "151674", "151675", "151676")#"151507", , "151671"

for (sample in list.samples) {
  input_dir = paste0("../data/", seq_name,"/",sample, "/export")
  pcs= read.table(paste(c(input_dir, "pcs.tsv"), collapse = '/'), header=T, sep='\t')
  locs= read.table(paste(c(input_dir, "locs.tsv"), collapse = '/'), header=T, sep='\t')
  W <- getSpatialNeighbors(locs, filterDist = 2)
  clusters <- getSpatiallyInformedClusters(pcs, W=W, k=20, alpha=1, beta=1)
  
  dir.output <-paste0("../output/", seq_name,"/",sample,"/MERINGUE");
  
  dir.create(dir.output, showWarnings = F)
  df = data.frame(clusters)
  write.table(df, file = file.path(dir.output, 'metadata.tsv'), sep='\t', quote=F, row.names = F, col.names = T)
}
