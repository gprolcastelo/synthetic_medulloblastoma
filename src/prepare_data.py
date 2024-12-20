import os, requests, gzip
import pandas as pd
import numpy as np

# URL of the file to download
url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE85217&format=file&file=GSE85217%5FM%5Fexp%5F763%5FMB%5FSubtypeStudy%5FTaylorLab%2Etxt%2Egz"

# Path to save the downloaded file
save_path = "data/raw/GEO"
os.makedirs(save_path, exist_ok=True)
filename = "GSE85217_M_exp_763_MB_SubtypeStudy_TaylorLab.txt.gz"
path_to_file = os.path.join(save_path, filename)

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Write the content to a file
    print('Downloading file...')
    with open(path_to_file, 'wb') as file:
        file.write(response.content)
    print(f"File downloaded successfully and saved to {path_to_file}")
else:
    print(f"Failed to download file. Status code: {response.status_code}")

# Extract .gz file
with gzip.open(path_to_file, 'rb') as f_in:
    with open(path_to_file.replace('.gz', ''), 'wb') as f_out:
        f_out.write(f_in.read())
# Delete .gz file
os.remove(path_to_file)

# Data downloaded directly from GEO and unzipped:
data_direct = pd.read_table(path_to_file.replace('.gz', ''), sep="\t", index_col=0)
print(data_direct.shape)

# Extract gene correspondence:
columns_genes = data_direct.columns[:4]
gene_correspondence = data_direct[columns_genes]
# Remove gene correspondence from data
data_direct = data_direct.drop(columns=columns_genes)
print(data_direct.shape, gene_correspondence.shape)

# Metadata obtained from the GEO website
metadata = pd.read_csv('data/raw/GEOquery/GSE85217_metadata.csv',index_col=0)
metadata.shape

# Set data patient names as index in metadata
new_column_names=metadata[metadata['title']==data_direct.columns].index
data_direct.columns = new_column_names

subtypes = metadata['subtype:ch1']
subtypes.name='Sample_characteristics_ch1'
subtypes.index.name = 'Sample_geo_accession'
subgroups = metadata['subgroup:ch1']
subgroups.name='Sample_characteristics_ch1'
print(subtypes.shape, subgroups.shape)

# Save data
data_direct.to_csv(os.path.join(save_path,'cavalli.csv'))
gene_correspondence.to_csv(os.path.join(save_path,'cavalli_gene_correspondence.csv'))
subtypes.to_csv(os.path.join(save_path,'cavalli_subtypes.csv'))
subgroups.to_csv(os.path.join(save_path,'cavalli_subgroups.csv'))