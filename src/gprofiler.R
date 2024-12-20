# Arguments
library(argparser)

# Create a parser
parse <- arg_parser("gprofiler enrichment analysis")

# Add arguments
parse <- add_argument(parse, "data_path", help = "Path to the input data file. This is a list of genes with ENSG IDs.", type = "character")
parse <- add_argument(parse, "save_path", help = "Path to save the output files.", type = "character")
# Parse the arguments
argv <- parse_args(parse)

# Use the data_path argument
data_path <- argv$data_path
save_path <- argv$save_path
if (!file.exists(save_path)) {
  dir.create(save_path, recursive = TRUE)
}

# Imports
library(gprofiler2)
library(ggplot2)
library(xtable)

today <- format(Sys.Date(), "%Y%m%d")


# Get list of SHAP genes
shap_genes <- read.csv(file=data_path, row.names = 1)


# Remove the final "_at" if it exists
genes <- sapply(shap_genes$selected_genes, function(x) sub("_at$", "", x))
length(genes)


# Use gprofiler on genes list
gostres <- gost(query = genes,
                organism = "hsapiens", significant = TRUE,
                correction_method = "fdr",
                domain_scope = "annotated",
                sources=c('GO','REAC','KEGG','WP')
                )

# A grid of barplots of the top 15 terms of each database

# Get the top 10 terms of each database
sources_used <- as.list(unique(gostres$result$source))
top_terms <- lapply(sources_used, function(x) {
  top_terms <- gostres$result[gostres$result$source == x,]
  top_terms <- top_terms[order(top_terms$p_value),]
  top_terms <- top_terms[1:20,]
  return(top_terms)
})

# Add log p-values to the top_terms data frames
for (i in 1:length(sources_used)) {
  top_terms_i <- top_terms[[i]]
  top_terms_i$log_p_value <- -log10(top_terms_i$p_value)
  top_terms[[i]] <- top_terms_i
}

# Plot the top 10 terms of each database
# Create a data frame for plotting
plot_data <- data.frame(
  source = character(),
  term_name = character(),
  p_value = numeric(),
  log_p_value = numeric()
)

# Populate the data frame with top terms
for (i in 1:length(sources_used)) {
  top_terms_i <- top_terms[[i]]
  plot_data <- rbind(plot_data, top_terms_i[, c("source", "term_name", "p_value","log_p_value")])
}

options(repr.plot.width=25, repr.plot.height=15)

# Check for duplicates, add database of origin to term name if duplicated
duplicated_terms <- duplicated(plot_data$term_name)
plot_data$term_name[duplicated_terms] <- paste(plot_data$term_name[duplicated_terms],' (', plot_data$source[duplicated_terms],' )')
# Plot the top terms using ggplot reorder(miRNA, -value)
plot_data$term_name <- factor(plot_data$term_name, levels = plot_data$term_name[order(plot_data$log_p_value)])

ggplot(plot_data, aes(x = log_p_value, y = term_name, fill = source)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ source, nrow = 3, scales = "free_y") +
  labs(x = "-log10(p-value)", y = "Term Name", title = "Top 10 Enriched Terms of Each Database") +
  theme_bw() +
  theme(
    axis.text.y = element_text(angle = 0, hjust = 1, size = 16),
    axis.title.x = element_text(size = 16, hjust = 0.1),
    axis.title.y = element_text(size = 16),
    plot.title = element_text(size = 20, hjust = 0.25)
  )

# Save the plot
ggsave(filename = paste0(save_path,'gostplot.pdf'), width = 25, height = 15, dpi=600)
ggsave(filename = paste0(save_path,'gostplot.png'), width = 25, height = 15, dpi=600)

# Save table with results
write.table(gostres$result[,1:13], file=paste0(save_path,'Enrichment_Analysis.csv'), sep=",", row.names=F)

