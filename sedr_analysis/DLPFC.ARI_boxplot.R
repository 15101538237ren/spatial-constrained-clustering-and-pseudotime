library(dplyr)
library(Giotto)
library(Seurat)
library(ggplot2)
library(patchwork)
library(ggthemes)
library(ggpubr)
library(mclust)
library(extrafont)
font_import()
options(bitmapType = 'cairo')

list.samples <- c("151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675", "151676")
list.methods <- c( "Seurat", "Giotto", "stLearn", "SpaGCN", "BayesSpace", "DGI", "DGI_spatial")

##### Generate data
c1 <- c()
c2 <- c()
c3 <- c()

for (sample in list.samples) {
  file.results <- file.path('./comparisons/', sample, '/Comparison/comparison.tsv')
  df.results <- read.table(file.results, sep='\t', header=T)
  for (method in list.methods){
    cluster <- df.results  %>% select(c(method))
    ARI <- adjustedRandIndex(x = df.results$layer_guess, y = cluster[, 1])
    
    c1 <- c(c1, method)
    c2 <- c(c2, sample)
    c3 <- c(c3, ARI)
  }
}

df.comp <- data.frame(method = c1,
                      sample = c2,
                      ARI = c3)

df.comp[df.comp$method=="DGI_spatial",1] = "SpaceFlow"

##### Plot results
df.comp$method <- as.factor(df.comp$method)
df.comp$method <- factor(df.comp$method, 
                         levels = c("SpaceFlow", "DGI", "BayesSpace", "SpaGCN", "Giotto", "stLearn", "Seurat"))#"VASC", "VASC_spatial", "GAE", "GAE_spatial", "VGAE", "VGAE_spatial", "DGI", "DGI_spatial", "APPNP", "GAT", "RGCN", "DGI_w1", "DGI_expr", "DGI_w_spatial", "DGI_expr_spatial", 

tapply(df.comp$ARI, df.comp$method, summary)

ggplot(df.comp, aes(x=method, y=ARI, fill=method)) + 
  geom_boxplot(width=0.5) + 
  geom_point(size = .2) +
  geom_jitter(width = 0.1, size=1) +
  coord_flip() +
  theme(
         legend.position="none",
        axis.title.y = element_blank(), 
        axis.text = element_text(size=14, color="black", family="Aial"),
        text=element_text(size=14, color="black", family="Aial"))

ggsave('../output/DLPFC/ARI_violin.svg', width=3, height=5)
