#devtools::install_github("mkuhn/dict")
library(CellChat)
library(patchwork)
library(reticulate)
source("filter_non_human_genes.R")
options(stringsAsFactors = FALSE)

data_dir = "../data/Visium/Chicken_Dev/"
fig_dir = "../output/chicken_bk/Valve/DGI_SP"
sample_name = "D10"
anndata_fp = paste(c(data_dir, "preprocessed", sample_name, "anndata_pp.h5ad"), collapse = '/')
annotation_fp = paste(c(data_dir, "preprocessed", sample_name, "cluster_anno.tsv"), collapse = '/')
color_fp = paste(c(data_dir, "putative_cell_type_colors", paste0(sample_name, ".csv",sep="")), collapse = '/')
ad <- import("anndata", convert = FALSE)
ad_object <- ad$read_h5ad(anndata_fp)
# access normalized data matrix
data.input <- t(py_to_r(ad_object$X))
data.input <- filter_non_human_genes(data.input)

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
cellchat@DB <- CellChatDB.use

# subset the expression data of signaling genes for saving computation cost
cellchat <- subsetData(cellchat) # This step is necessary even if using the whole database
future::plan("multiprocess", workers = 4) # do parallel
cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)
# project gene expression data onto PPI network (optional)
# cellchat <- projectData(cellchat, PPI.human)

#Compute the communication probability and infer cellular communication network
cellchat <- computeCommunProb(cellchat, population.size = F)
# Filter out the cell-cell communication if there are only few number of cells in certain cell groups
cellchat <- filterCommunication(cellchat, min.cells = 10)

#Infer the cell-cell communication at a signaling pathway level
cellchat <- computeCommunProbPathway(cellchat)

#Calculate the aggregated cell-cell communication network
cellchat <- aggregateNet(cellchat)

pathways.sig.all <- cellchat@netP$pathways
identity_df <- data.frame(table(cellchat@idents))
colors_for_use <- colors_for_cell_types[match(identity_df$Var1, colors_df$Abbr)]
groupSize <- as.numeric(table(cellchat@idents))
valve_cells_numb <- c(16, 17)
non_valve <- c(2,3,8,11,13)

pathways.show <- c("MK") 
par(mfrow=c(1,1))

pairLR.MK <- extractEnrichedLR(cellchat, signaling = pathways.show, geneLR.return = FALSE)

netVisual_aggregate(cellchat, signaling = pathways.show, layout = "circle")
LR.show = c("MDK_SDC2")
netVisual_individual(cellchat, signaling = pathways.show,
                     sources.use= c("Valve-1","Valve-2"), targets.use=c(""), pairLR.use = LR.show, layout = "circle")


pdf(paste(c(fig_dir, "ccc_send_by_valve_PTN.pdf"), collapse = '/'), width=7, height=4, pointsize=12)
netVisual_chord_gene(cellchat, sources.use = valve_cells_numb, targets.use = non_valve, color.use=colors_for_use, signaling = c("PTN"), lab.cex = 0.5,legend.pos.y = 30, title.name = "Cell Cell signaling\nfrom valve cells")#
dev.off()

pdf(paste(c(fig_dir, "ccc_recv_by_valve_PTN.pdf"), collapse = '/'), width=7, height=4, pointsize=14)
netVisual_chord_gene(cellchat, sources.use = non_valve, targets.use = valve_cells_numb, color.use=colors_for_use, signaling = c("PTN"), lab.cex = 0.5,legend.pos.y = 30, title.name = "Cell Cell signaling\nto valve cells")
dev.off()
netVisual_bubble(cellchat, sources.use = valve_cells_numb , targets.use = non_valve, signaling = c("MK", "PTN", "CXCL", "ANGPT"), remove.isolate = T, thresh = 0.01)
netVisual_bubble(cellchat, sources.use = non_valve, targets.use = valve_cells_numb, signaling = c("MK", "PTN", "CXCL", "ANGPT"), remove.isolate = T, thresh = 0.01)



library(NMF)
library(ggalluvial)
nPatterns = 4
cellchat <- identifyCommunicationPatterns(cellchat, pattern = "outgoing", k = nPatterns)
netAnalysis_river(cellchat, pattern = "outgoing")

nPatterns = 3
cellchat <- identifyCommunicationPatterns(cellchat, pattern = "incoming", k = nPatterns)
netAnalysis_river(cellchat, pattern = "incoming")
