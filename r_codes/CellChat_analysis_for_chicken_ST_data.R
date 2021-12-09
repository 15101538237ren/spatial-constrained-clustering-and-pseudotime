library(CellChat)
library(patchwork)
library(reticulate)
source("filter_non_human_genes.R")
options(stringsAsFactors = FALSE)

data_dir = "../data/Visium/Chicken_Dev/preprocessed/"
sample_name = "D14"
anndata_fp = paste(c(data_dir, sample_name, "anndata_pp.h5ad"), collapse = '/')
annotation_fp = paste(c(data_dir, sample_name, "cluster_anno.tsv"), collapse = '/')

ad <- import("anndata", convert = FALSE)
ad_object <- ad$read_h5ad(anndata_fp)

# access normalized data matrix
data.input <- t(py_to_r(ad_object$X))
data.input <- filter_non_human_genes(data.input)

# access meta data
meta <- read.table(annotation_fp, header=T, sep='\t')
rownames(meta) <- meta$Cell

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
cellchat <- computeCommunProb(cellchat)
# Filter out the cell-cell communication if there are only few number of cells in certain cell groups
cellchat <- filterCommunication(cellchat, min.cells = 10)

#Infer the cell-cell communication at a signaling pathway level
cellchat <- computeCommunProbPathway(cellchat)

#Calculate the aggregated cell-cell communication network
cellchat <- aggregateNet(cellchat)

pathways.sig.all <- cellchat@netP$pathways
identity_df <- data.frame(table(cellchat@idents))
groupSize <- as.numeric(table(cellchat@idents))
valve_cells_numb <- c(12:15)
MT_valve_cells_numb <- c(11)
others <- c(1:10)
#non_MT <- c(others, valve_cells_numb)
netVisual_chord_gene(cellchat, sources.use = valve_cells_numb, targets.use = MT_valve_cells_numb, lab.cex = 0.5,legend.pos.y = 30)
netVisual_chord_gene(cellchat, sources.use = MT_valve_cells_numb, targets.use = valve_cells_numb, lab.cex = 0.5,legend.pos.y = 30)
netVisual_chord_gene(cellchat, sources.use = MT_valve_cells_numb, targets.use = valve_cells_numb, lab.cex = 0.5,legend.pos.y = 30)



