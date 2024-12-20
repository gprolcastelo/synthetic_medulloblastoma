# This script is used to cluster the g3g4 data using the k-means algorithm.

# Load the argparse library
library(argparse)

# Create a parser object
parser <- ArgumentParser(description = 'Cluster the G3 and G4 data using the k-means algorith')

# Add arguments
parser$add_argument('--data_path', type = 'character', help = 'Path to data')
parser$add_argument('--metadata_path',type = 'character', help = 'Path to metadata')
parser$add_argument('--clusterAlg', type = 'character', nargs = '+', help = 'List of strings indicating the clustering algorithm(s) to use')
parser$add_argument('--real_space', action = 'store_true', help = 'Boolean flag that is TRUE if used and FALSE otherwise')
parser$add_argument('--save_path', type = 'character', help = 'Path to save results')
parser$add_argument('--plot_ext', type = 'character', default = 'pdf', help = 'Extension to save plots')
parser$add_argument('--reps', type = 'integer', default = 1000, help = 'Number of repetitions')
parser$add_argument('--psamples', type = 'numeric', default = 0.8, help = 'Proportion of items to sample')
parser$add_argument('--pfeatures', type = 'numeric', default = 1, help = 'Proportion of features to sample')
parser$add_argument('--maxk', type = 'integer', default = 3, help = 'Maximum number of clusters')



# Parse the arguments
args <- parser$parse_args()

# Example uses:
# With real data:
# Rscript src/g3g4_clustering.R --data_path "data/interim/20240729_consensusclustering/rnaseq_g3g4_noprepro.csv" \
#                               --clusterAlg "hc" "km" "pam" \
#                               --real_space \
#                               --save_path "data/interim/20240729_consensusclustering/results" \
#                               --plot_ext "pdf" \
#                               --reps 1000 \
#                               --psamples 0.8 \
#                               --pfeatures 1 \
#                               --maxk 3
#
# With latent data:
# Rscript src/g3g4_clustering.R --data_path "data/interim/20240729_consensusclustering/df_z_g3g4.csv" \
#                               --clusterAlg "hc" "km" "pam" \
#                               --save_path "data/interim/20240729_consensusclustering/results" \
#                               --plot_ext "pdf" \
#                               --reps 1000 \
#                               --psamples 0.8 \
#                               --pfeatures 1 \
#                               --maxk 3
library(ConsensusClusterPlus)

ensure_directory <- function(path) {
  if (!dir.exists(path)) {
    print(paste0("Creating directory: ", path))
    dir.create(path, recursive = TRUE)
  }
}
# Load the data
real_space <- args$real_space # Use real space data
#rnaseq <- FALSE # Use latent data

# if (rnaseq){
#   print("Using RNA-seq data")
#   data_path <- "../data/interim/20240729_consensusclustering/rnaseq_g3g4_noprepro.csv"
#   #data_path <- "data/interim/20240729_consensusclustering/rnaseq_g3g4.csv"
#   clusterAlg <- c("hc") # Clustering algorithms to use
# } else {
#   print("Using latent data")
#   data_path <- "../data/interim/20240729_consensusclustering/df_z_g3g4.csv"
#   clusterAlg <- c("hc", "km","pam") # Clustering algorithms to use
# }
#data_path <- "data/interim/20240729_consensusclustering/df_z_g3g4.csv"
#rnaseq <- TRUE
#data_path <- "data/interim/20240729_consensusclustering/rnaseq_g3g4_noprepro.csv"
# Path to save consenus clustering results
title <- args$save_path
# Extension to save plots
plot_ext <- args$plot_ext
reps <- args$reps # Number of repetitions
proportion <- args$psamples # Proportion of items to sample
pFeature <- args$pfeatures # Proportion of features to sample

maxK <- args$maxk # Maximum number of clusters
# Run
data <- read.table(args$data_path, header=TRUE, sep=",", row.names=1)
metadata <- read.table(args$metadata_path, header=TRUE, sep=",", row.names=1)
print('metadata head:')
head(metadata)
# Check if samples coincide:
print("Checking if number of samples coincide")
if (dim(data)[2] != dim(metadata)[1]){
  data <- t(data)
}
if (dim(data)[2] != dim(metadata)[1]){
  stop("Number of samples do not coincide")
}

# Check for missing data
print("Checking for missing data")
print(sum(is.na(data)))

# Select columns corresponding only Groups 3 and 4
patients <- rownames(metadata)
patients_g3g4 <- patients[metadata$Sample_characteristics_ch1 %in% c("Group3", "Group4")]
data <- data[,patients_g3g4]
print("Data shape after filtering:")
dim(data)
# Convert data to matrix
data <- as.matrix(data)

print("Data shape before selecting top 10,000 most informative genes:")
dim(data)
# head(data)

if (real_space){
  print("Using real data")
  # Select top 10,000 most informative genes
  mads <- apply(data, 1, mad)
  data <- data[rev(order(mads))[1:10000],]
  print("Data shape after electing top 10,000 most informative genes:")
  dim(data)
} else {
  print("Using latent data")
}

# Ensure the base directory exists
ensure_directory(title)

# Run ConsensusClusterPlus:
for (cluster_i in args$clusterAlg){
  print(paste0("Running ConsensusClusterPlus with ", cluster_i))
  title_i <- paste0(title, "/", cluster_i)
  ensure_directory(title_i)  # Ensure the directory exists
  print(title_i)
  results <- ConsensusClusterPlus(data, maxK=maxK, reps=reps, pItem=proportion,
                                  pFeature=pFeature, title=title_i, clusterAlg=cluster_i,
                                  distance="pearson",
                                  seed=2023, plot=plot_ext,
                                  writeTable=FALSE)
  # calculate clusterconsensus and item-consensus results
  icl <- calcICL(results, title=title_i, plot=plot_ext,writeTable=FALSE)
  # Save results to files
  # From ConsensusClusterPlus
  print("Saving ConsensusClusterPlus results")
  for (k in 2:maxK){
    # Save all consensus matrices: from k=2 to k=maxK
    consensusMatrix <- results[[k]][["consensusMatrix"]]
    print(paste0("Saving consensusMatrix_k", k))
    write.table(consensusMatrix, file=paste0(title_i, "/consensusMatrix_k", k, ".csv"), sep=",")
    # Save all consensus trees: from k=2 to k=maxK
    # consensusTree <- results[[k]][["consensusTree"]]
    # print(paste0("Saving consensusTree_k", k))
    # write.table(consensusTree, file=paste0(title_i, "/consensusTree_k", k, ".csv"), sep=",")
    # Save all consensus classes: from k=2 to k=maxK
    print(paste0("Saving consensusClass_k", k))
    consensusClass <- results[[k]][["consensusClass"]]
    write.table(consensusClass, file=paste0(title_i, "/consensusClass_k", k, ".csv"), sep=",")
  }
  # From calcICL
  write.table(icl[["clusterConsensus"]], file=paste0(title_i, "/clusterConsensus.csv"), sep=",")
  write.table(icl[["itemConsensus"]], file=paste0(title_i, "/itemConsensus.csv"), sep=",")
}

# results <- ConsensusClusterPlus(data, maxK=6, reps=reps, pItem=proportion,
#                                 pFeature=1, title=title, clusterAlg="hc",
#                                 distance="pearson",
#                                 seed=2023, plot=plot_ext,writeTable=TRUE)

#For .example, the top five rows and columns of results for k=2:
# results[[2]][["consensusMatrix"]][1:5,1:5]
#
# #consensusTree - hclust object
# results[[2]][["consensusTree"]]
#
#  #consensusClass - the sample classifications
# results[[2]][["consensusClass"]][1:5]

# calculate clusterconsensus and item-consensus results
# icl <- calcICL(results, title=title, plot=plot_ext)
# icl[["clusterConsensus"]]
# icl[["itemConsensus"]][1:5,]

