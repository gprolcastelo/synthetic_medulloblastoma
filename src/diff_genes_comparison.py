import mygene, os
import pandas as pd
import numpy as np

def load_data(path_external, path_internal_differential):
    genes_external = pd.read_csv(path_external, sep=' ')
    genes_internal_diff = pd.read_csv(path_internal_differential, index_col=0)
    return genes_external, genes_internal_diff

def diff_genes_in_cluster(df_external,group_of_interest='G3_G4'):
    '''
    Get set of genes that are in the specified cluster group_of_interest but not in the rest
    :param df_external: DataFrame with genes and clusters. Minimum columns: Genes, Cluster 
    :param group_of_interest: Cluster of interest, i.e., group whose genes we want to compare with the rest. Default: 'G3_G4'
    :return: List of genes that are in the specified cluster group_of_interest but not in the rest
    '''
    # Number of patients in the group of interest
    num_pats_interest = len(df_external[df_external['Cluster']==group_of_interest])
    print(f'Number of patients in {group_of_interest} cluster:'+'\t'*4, num_pats_interest)
    # Unique genes in the group of interest
    genes_interest=[df_external[df_external['Cluster']==group_of_interest]['Genes'].values[i].split('_') for i in range(num_pats_interest)]
    unique_interest = np.unique([gene for sublist in genes_interest for gene in sublist])
    print(f'Number of unique genes in {group_of_interest} cluster:'+'\t'*2,len(unique_interest))
    # Number of patients in the rest
    num_pats_rest = len(df_external[df_external['Cluster']!=group_of_interest])
    print('Number of patients in the rest:'+'\t'*7, num_pats_rest)
    # Unique genes in the rest
    genes_rest = [df_external[df_external['Cluster']!=group_of_interest]['Genes'].values[i].split('_') for i in range(num_pats_rest)]
    unique_rest = np.unique([gene for sublist in genes_rest for gene in sublist])
    print('Number of unique genes in all other clusters:\t',len(unique_rest))
    # Genes in the group of interest but not in the rest
    diff_genes = np.setdiff1d(unique_interest, unique_rest, assume_unique=True).tolist()
    print(f'Number of genes in {group_of_interest} but not in the rest:\t',len(diff_genes))
    return diff_genes

def gene_equivalences(list_genes):
    mg = mygene.MyGeneInfo()
    # Get entrez ids, ensemble ids and gene symbols for the genes of interest from the external dataset
    df_equivalences = mg.getgenes(list_genes, fields="entrezgene,ensembl.gene,symbol",as_dataframe=True)
    return df_equivalences


def important_genes_northcott2019(list_compare_genes):
    # Define the lists for each group
    WNT_genes = ["CTNNB1", "DDX3X", "SMARCA4", "CSNK2B", "TP53", "KMT2D", "PIK3CA", "BAI3", "EPHA7", "ARID1A", "ARID2", "SYNCRIP", "ATM"]
    SHH_genes = ["TERT", "DDX3X", "KMT2D", "CREBBP", "TCF4", "PTEN", "KMT2C", "FBXW7", "GSE1", "BCOR", "PRKAR1A", "IDH1"]
    Group_3_genes = ["MYC", "SMARCA4", "KBTBD4", "CTDNEP1", "KMT2D", "MYCN", "OTX2", "OTX2", "CDK6", "GFI1", "GFI1B"]
    Group_4_genes = ["PRDM6", "SNCAIP", "KDM6A", "ZMYM3", "KMT2C", "KBTBD4", "MYCN"]

    # Pad the lists with None to make them the same length
    max_length = max(len(WNT_genes), len(SHH_genes), len(Group_3_genes), len(Group_4_genes))
    WNT_genes += [None] * (max_length - len(WNT_genes))
    SHH_genes += [None] * (max_length - len(SHH_genes))
    Group_3_genes += [None] * (max_length - len(Group_3_genes))
    Group_4_genes += [None] * (max_length - len(Group_4_genes))

    important_genes = pd.DataFrame({
        'WNT': WNT_genes,
        'SHH': SHH_genes,
        'Group_3': Group_3_genes,
        'Group_4': Group_4_genes
    })

    # Get the genes that are in the important genes list
    # Filter the genes in each group
    filtered_genes = {
        'WNT': important_genes['WNT'][important_genes['WNT'].isin(list_compare_genes)],
        'SHH': important_genes['SHH'][important_genes['SHH'].isin(list_compare_genes)],
        'Group_3': important_genes['Group_3'][important_genes['Group_3'].isin(list_compare_genes)],
        'Group_4': important_genes['Group_4'][important_genes['Group_4'].isin(list_compare_genes)]
    }

    # Convert the dictionary to a DataFrame
    filtered_genes_df = pd.DataFrame(dict([(k, pd.Series(v.values)) for k, v in filtered_genes.items()]))

    return important_genes, filtered_genes_df

def main(args):
    # Load data from external and internal genes datasets
    genes_external, genes_internal_diff = load_data(
        args.external_genes,
        args.internal_genes_differential)

    # Get genes that are in the G3_G4 cluster but not in the rest
    diff_genes_g3g4 = diff_genes_in_cluster(df_external=genes_external, group_of_interest=args.group_of_interest)
    # Get gene equivalences for the genes from the external dataset
    df_equivalences_external = gene_equivalences(diff_genes_g3g4)
    # Get gene equivalences for the genes from the internal dataset
    entrez_genes_internal = [i.removesuffix('_at') for i in genes_internal_diff['genes'].to_list()]
    entrez_genes_internal = pd.Series(entrez_genes_internal, name='genes')
    genes_internal_equivalences = gene_equivalences(entrez_genes_internal)
    # Check if the genes from the external dataset are in the internal datasets
    coincidences_df = df_equivalences_external[df_equivalences_external['ensembl.gene'].isin(entrez_genes_internal)]
    print('Number of genes in the external dataset that are also in the internal differential dataset:')
    print(coincidences_df.shape[0])
    # Get the important genes from Northcott et al., 2019, and check coincidences on the internal dataset
    important_genes, filtered_genes_df = important_genes_northcott2019(genes_internal_equivalences['symbol'])
    print('Important genes from Northcott et al., 2019 that coincide with internal set:')
    print(filtered_genes_df)
    # Save data
    os.makedirs(args.save_path, exist_ok=True)
    df_equivalences_external.to_csv(os.path.join(args.save_path, 'external_genes_equivalences.csv'))
    genes_internal_equivalences.to_csv(os.path.join(args.save_path, 'internal_genes_equivalences.csv'))
    coincidences_df.to_csv(os.path.join(args.save_path, 'coincidences_with_external.csv'))
    important_genes.to_csv(os.path.join(args.save_path, 'important_genes_northcott2019.csv'))
    filtered_genes_df.to_csv(os.path.join(args.save_path, 'coincidences_with_northcott2019.csv'))
    print('Results saved to:', args.save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Comparing genes from external and internal datasets')
    parser.add_argument('--external_genes', type=str, help='Path to external genes')
    parser.add_argument('--internal_genes_differential', type=str, help='Path to internal genes that are differentially expressed in the cluster of interest')
    # parser.add_argument('--internal_genes_shap', type=str, help='Path to internal genes')
    parser.add_argument('--group_of_interest', type=str, help='Cluster/group of interest', default='G3_G4')
    parser.add_argument('--save_path', type=str,
                        help='Path to save the results')

    # Simulate command-line arguments
    # import sys
    # sys.argv = ['notebook',  # notebook simulates the script name, then the arguments
    #             '--external_genes', 'data/external/Supplementary_Table_5.csv',
    #             '--internal_genes_differential','data/processed/20241115_differentially_expressed_genes/synth_patients/always_diff_genes.csv',
    #             '--group_of_interest', 'G3_G4',
    #             '--save_path', 'data/processed/20241115_genes_comparison/synth_patients/',
    #             ]
    args = parser.parse_args()
    main(args)

# Run the following in the console:

# a) Comparing synth data
# python src/diff_genes_comparison.py --external_genes data/external/Supplementary_Table_5.csv \
#                                 --internal_genes_differential data/processed/20241115_differentially_expressed_genes/synth_patients/always_diff_genes.csv \
#                                 --group_of_interest G3_G4 \
#                                 --save_path data/processed/20241115_genes_comparison/synth_patients
# b) Comparing real data
# python src/diff_genes_comparison.py --external_genes data/external/Supplementary_Table_5.csv \
#                                 --internal_genes_differential data/processed/20241115_differentially_expressed_genes/real_patients/always_diff_genes.csv \
#                                 --group_of_interest G3_G4 \
#                                 --save_path data/processed/20241115_genes_comparison/real_patients




# len(genes_external)
#
# all_genes_external=[genes_external['Genes'].values[i].split('_') for i in range(len(genes_external))]
# print(len(all_genes_external))
# unique_external = np.unique([gene for sublist in all_genes_external for gene in sublist])
# print(len(unique_external))
#
# df_equivalences_external = mg.getgenes(unique_external, fields="entrezgene,ensembl.gene,symbol",as_dataframe=True)
#
#
# df_equivalences_external['symbol'].isna().sum()
#
#
#
# np.isin(df_equivalences_external['symbol'].tolist(), genes_internal_shap_list).sum()
#
# np.isin(genes_internal_shap_list, df_equivalences_external['symbol'].tolist()).sum()