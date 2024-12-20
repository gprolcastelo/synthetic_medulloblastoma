import os
import pandas as pd
from src.visualization.visualize import plot_umap

def load_data(data_path,metadata_path,consensus_path):
    data = pd.read_csv(data_path,index_col=0)
    metadata = pd.read_csv(metadata_path,index_col=0).replace({'Group3':'Group 3','Group4':'Group 4'})
    metadata=pd.Series(metadata.values.flatten(), index=metadata.index, name='group')
    consensus = pd.DataFrame()
    for k_here in range(2, 4):
        consensus_i = pd.read_csv(os.path.join(consensus_path, f'consensusClass_k{k_here}.csv'))
        consensus[f'k_{k_here}'] = consensus_i['x']
    return data, metadata, consensus

def get_consensus(consensus, metadata):
    consensus_k2 = metadata.copy()
    consensus_k2.loc[consensus.index] = consensus['k_2']
    consensus_k3 = metadata.copy()
    consensus_k3.loc[consensus.index] = consensus['k_3']
    return consensus_k2, consensus_k3

def find_overlap(metadata, consensus_k2, consensus_k3):
    # According to Cavalli, to find the overlap, they
    # "counted the number of samples that were initially considered to be of a particular
    # subgroup for k=2 and moved to be in another subgroup at k=3"
    # For k=2, 1: Group 3, 2: Group 4
    k2_dict = {1: 'Group 3', 2: 'Group 4'}
    consensus_k2.replace(k2_dict, inplace=True)
    # For k=3, 2 and 3: Group 4, 1: Group 3
    k3_dict = {2: 'Group 4', 3: 'Group 4', 1: 'Group 3'}
    consensus_k3.replace(k3_dict, inplace=True)
    # Contingency table between k=2 and k=3
    cross_tab_k2_k3=pd.crosstab(consensus_k2, consensus_k3, margins=True)
    cross_tab_k2_k3.index.name = 'k=2'
    cross_tab_k2_k3.columns.name = 'k=3'
    # Get the patients that have changed group from consensus_k2_clinical to consensus_k3
    changed_patients_k3_to_k2 = consensus_k2[consensus_k2 != consensus_k3].index
    metadata_changed_k3_to_k2 = metadata.copy()
    metadata_changed_k3_to_k2[changed_patients_k3_to_k2] = 'G3-G4'
    # Get corresponding contingency table
    original_consensus_comparison = pd.crosstab(metadata, metadata_changed_k3_to_k2, margins=True)
    original_consensus_comparison.index.name = 'Original'
    original_consensus_comparison.columns.name = 'ConsensusClustering'
    return cross_tab_k2_k3, metadata_changed_k3_to_k2, original_consensus_comparison
def main(args):
    # Load data
    data, metadata, consensus = load_data(data_path=args.data_path,consensus_path=args.consensus_path,metadata_path=args.metadata_path)
    # Get group assignments for k=2 and k=3 from the consensus clustering results
    consensus_k2, consensus_k3 = get_consensus(consensus=consensus, metadata=metadata)
    # Plot umaps for k=2 and k=3
    dict_umap_consensus_k2 = {'SHH': '#b22222', 'WNT': '#6495ed', 1: '#ffd700', 2: '#008000'}
    dict_umap_consensus_k3 = {'SHH': '#b22222', 'WNT': '#6495ed', 1: '#ffd700', 2: '#008000', 3: '#ff69b4'}
    plot_umap(data, consensus_k2, dict_umap_consensus_k2, n_components=2,save_fig=True,
              # save_as=os.path.join(args.save_path, 'k2_latent' if args.use_latent else 'k2_noprepro'),
              save_as=os.path.join(args.save_path, 'k2'),
              seed=2023, title=None,show=False)
    plot_umap(data, consensus_k3, dict_umap_consensus_k3, n_components=2, save_fig=True,
              # save_as=os.path.join(args.save_path, 'k3_latent' if args.use_latent else 'k3_noprepro'),
              save_as=os.path.join(args.save_path, 'k3'),
              seed=2023, title=None,show=False)
    # Find overlap between k=2 and k=3
    # if args.use_latent:
    cross_tab_k2_k3, metadata_changed_k3_to_k2, original_consensus_comparison = find_overlap(
        metadata=metadata, consensus_k2=consensus_k2, consensus_k3=consensus_k3)
    cross_tab_k2_k3.to_csv(os.path.join(args.save_path, 'contingency_k2_k3.csv'))
    cross_tab_k2_k3.to_latex(os.path.join(args.save_path, 'contingency_k2_k3.tex'))
    original_consensus_comparison.to_csv(os.path.join(args.save_path, 'original_consensus_comparison.csv'))
    original_consensus_comparison.to_latex(os.path.join(args.save_path, 'original_consensus_comparison.tex'))
    metadata_changed_k3_to_k2.to_csv(os.path.join(args.save_path, 'metadata_changed_k3_to_k2.csv'))
    # Plot umap with In between groups:
    dict_umap = {'SHH': '#b22222', 'WNT': '#6495ed', 'Group 3': '#ffd700', 'Group 4': '#008000', 'G3-G4': '#db7093'}
    # Replace literal \n with an actual newline character
    title = args.title.replace('\\n', '\n')
    plot_umap(data, metadata_changed_k3_to_k2, dict_umap, n_components=2, save_fig=True,
              # save_as=os.path.join(args.save_path, 'k3_to_k2_latent' if args.use_latent else 'k3_to_k2_noprepro'),
              save_as=os.path.join(args.save_path, 'k3_to_k2'),
              seed=2023, title=title,show=False)
    return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze results from ConsensusCluster')
    parser.add_argument('--data_path', type=str, help='Path to the data file')
    parser.add_argument('--metadata_path', type=str, help='Path to the metadata file')
    parser.add_argument('--consensus_path', type=str, help='Path to the directory containing the ConsensusCluster results')
    parser.add_argument('--save_path', type=str, help='Path to the directory to save the results')
    # parser.add_argument('--use_latent', action='store_true', help='Use the latent space instead of the original data')
    parser.add_argument('--title', type=str, help='Title of the UMAP', default='')
    args = parser.parse_args()
    main(args)

# Use with:
# python src/consensuscluster_analysis.py --data_path data/interim/20240301_Mahalanobis/cavalli.csv \
#                                                                     --metadata_path data/cavalli_subgroups.csv \
#                                                                     --consensus_path data/interim/20240729_consensusclustering/results_rnaseq_noprepro/hc/ \
#                                                                     --save_path reports/figures/20241023_consensusclustering_umaps \

# Or with the latent space:
# python src/consensuscluster_analysis.py --data_path data/interim/20240301_Mahalanobis/cavalli.csv \
#                                                                     --metadata_path data/cavalli_subgroups.csv \
#                                                                     --consensus_path data/interim/20240729_consensusclustering/results_latent/km/ \
#                                                                     --save_path reports/figures/20241023_consensusclustering_umaps \
#                                                                     --use_latent