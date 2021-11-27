BASE_DIR=/Users/emmanueldollinger/PycharmProjects/spatial-constrained-clustering-and-pseudotime/sedr_analysis
cd $BASE_DIR

conda activate stlearn

declare -a samples=('151507' '151508' '151509' '151510' '151669' '151670' '151672' '151673' '151674' '151675' '151676' '151671')
declare -a clusters=(7 7 7 7 5 5 5 7 7 7 7 5)

for i in "${!samples[@]}"; do 
	sample="${samples[i]}"
	n_clusters="${clusters[i]}"
  	echo "sample: "$sample " cluster:"$n_clusters
  	#python DLPFC_SpaGCN.py $sample $n_clusters &
	python DLPFC_stLearn.py $sample &
	# Rscript DLPFC_Seurat.R $sample $n_clusters
	# Rscript DLPFC_BayesSpace.R $sample $n_clusters
	#Rscript DLPFC_Giotto.R $sample $n_clusters
	#Rscript DLPFC_comp.R $sample
done

#Rscript DLPFC.ARI_boxplot.R




BASE_DIR=/Users/emmanueldollinger/PycharmProjects/spatial-constrained-clustering-and-pseudotime
SOURCE_DIR=/Users/emmanueldollinger/PycharmProjects/SEDR
cd $BASE_DIR

declare -a samples=('151671') #'151507' '151508' '151509' '151510' '151669' '151670' '151672' '151673' '151674' '151675' '151676' 

for i in "${!samples[@]}"; do 
	sample="${samples[i]}"
  	echo "sample: "$sample
  	cp -r $SOURCE_DIR/output/DLPFC/$sample/Giotto $BASE_DIR/output/DLPFC/$sample
done
