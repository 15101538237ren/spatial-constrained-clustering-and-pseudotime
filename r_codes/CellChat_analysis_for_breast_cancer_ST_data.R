#devtools::install_github("mkuhn/dict")
library(CellChat)
library(patchwork)
library(reticulate)
options(stringsAsFactors = FALSE)

data_dir = "../data/Visium/breast_cancer/"
fig_dir = "../output/breast_cancer/G1/DGI_SP"
sample_name = "G1"
anndata_fp = paste(c(data_dir, "preprocessed", sample_name, "anndata_pp.h5ad"), collapse = '/')
annotation_fp = paste(c(data_dir, "preprocessed", sample_name, "cluster_anno.tsv"), collapse = '/')
color_fp = paste(c(data_dir, "putative_cell_type_colors", paste0(sample_name, ".csv",sep="")), collapse = '/')
ad <- import("anndata", convert = FALSE)
ad_object <- ad$read_h5ad(anndata_fp)
# access normalized data matrix
data.input <- t(py_to_r(ad_object$X))
rownames(data.input) <- rownames(py_to_r(ad_object$var))
colnames(data.input) <- rownames(py_to_r(ad_object$obs))

# access meta data
meta <- read.table(annotation_fp, header=T, sep='\t')
rownames(meta) <- meta$Cell
colors_df <- read.table(color_fp, header=T, sep=',')
colors_for_cell_types = paste("#", colors_df$Color, sep="")

# coerce a dgRMatrix to a dgCMatrix
setAs("dgRMatrix", to = "dgCMatrix", function(from){
  as(as(from, "CsparseMatrix"), "dgCMatrix")
})


#Create a CellChat object using data matrix as input

cellchat <- createCellChat(object = data.input, meta = meta, group.by = "Annotation")
CellChatDB <- CellChatDB.human # use CellChatDB.mouse if running on mouse data
CellChatDB.use <- subsetDB(CellChatDB, search = "Secreted Signaling") # use Secreted Signaling
cellchat@DB <- CellChatDB

# subset the expression data of signaling genes for saving computation cost
cellchat <- subsetData(cellchat) # This step is necessary even if using the whole database
# future::plan("multiprocess", workers = 4) # do parallel
cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)
# project gene expression data onto PPI network (optional)
# cellchat <- projectData(cellchat, PPI.human)

#Compute the communication probability and infer cellular communication network
cellchat <- computeCommunProb(cellchat, population.size = F)
# Filter out the cell-cell communication if there are only few number of cells in certain cell groups
cellchat <- filterCommunication(cellchat, min.cells = 5)

#Infer the cell-cell communication at a signaling pathway level
cellchat <- computeCommunProbPathway(cellchat)

#Calculate the aggregated cell-cell communication network
cellchat <- aggregateNet(cellchat)

pathways.sig.all <- cellchat@netP$pathways
identity_df <- data.frame(table(cellchat@idents))
colors_for_use <- colors_for_cell_types[match(identity_df$Var1, colors_df$Abbr)]
groupSize <- as.numeric(table(cellchat@idents))
invasive1 <- c(6)
inv2n_intf3 <- c(7,9)

pdf(paste(c(fig_dir, "ccc_send_by_invasive1.pdf"), collapse = '/'), width=7, height=5, pointsize=12)
netVisual_chord_gene(cellchat, sources.use = invasive1, targets.use = inv2n_intf3,signaling=c("COLLAGEN"), color.use=colors_for_use, lab.cex = 0.5,legend.pos.y = 30, title.name = "L-R interactions from Invasive-1")#"FN1", "COLLAGEN", "ANGPTL", "THBS", "PERIOSTIN", "APP", "CD99"
dev.off()


imm1 <- c(3)
others <- c(1, 4, 8)
pdf(paste(c(fig_dir, "ccc_send_by_immune_reg1.pdf"), collapse = '/'), width=7, height=5, pointsize=12)
netVisual_chord_gene(cellchat, sources.use = imm1, targets.use = others,signaling=c("MK", "APP"), color.use=colors_for_use, lab.cex = 0.5,legend.pos.y = 30, title.name = "L-R interactions from Imm-Reg-1")#
dev.off()


library(NMF)
library(ggalluvial)
nPatterns = 4
cellchat <- identifyCommunicationPatterns(cellchat, pattern = "outgoing", k = nPatterns)
netAnalysis_river(cellchat, pattern = "outgoing")

nPatterns = 3
cellchat <- identifyCommunicationPatterns(cellchat, pattern = "incoming", k = nPatterns)
netAnalysis_river(cellchat, pattern = "incoming")
