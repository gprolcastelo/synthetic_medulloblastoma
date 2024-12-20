library(GEOquery)
accession = "GSE85217"
# Create new string that includes the accession number
savepath = "data/raw/GEOquery/"
# Create the directory if it does not exist
if (!dir.exists(savepath)) {
  dir.create(savepath, recursive = TRUE)
  print(paste("Directory created:", savepath))
} else {
  print("Directory already exists")
}
filename = paste0(accession, "_series_matrix.txt.gz")
path_to_file = paste0(savepath, filename)
# Download the data
gse <- getGEO(accession,GSEMatrix=TRUE,destdir=savepath)

# Check if gse is a list and extract the first element if it is
if (is.list(gse)) {
  gse <- gse[[1]]
}

# Extract the expression data
expression_data <- exprs(gse)

# Extract the metadata
metadata <- pData(gse)

# Save the data as a csv file
write.csv(expression_data, file = paste0("data/raw/GEOquery/", accession, "_expression_data.csv"), row.names = TRUE)
write.csv(metadata, file = paste0("data/raw/GEOquery/", accession, "_metadata.csv"), row.names = TRUE)