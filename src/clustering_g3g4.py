import os, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.neighbors import kneighbors_graph
from src.visualization.visualize import plot_umap
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from src.models.train_model import set_seed
from src.models.my_model import VAE
from src.adjust_reconstruction import NetworkReconstruction as model_net_here
from src.data_processing.classification import *
from src.utils import apply_VAE, get_hyperparams
plt.style.use('ggplot')

def load_data(data_path, metadata_path):
    # Load data
    data = pd.read_csv(data_path, index_col=0)
    metadata = pd.read_csv(metadata_path, index_col=0)
    metadata = metadata['Sample_characteristics_ch1']
    metadata.replace({'Group3': 'Group 3', 'Group4': 'Group 4'}, inplace=True)
    # Check if the data and metadata have the same number of samples
    # in at least one of the dimensions, and
    # make sure that the data is in the right shape (samples x features)
    print('Data shape:', data.shape)
    print('Metadata shape:', metadata.shape)
    if data.shape[0]!=metadata.shape[0]:
        print('Transposing the data')
        data = data.T
        print('Data shape after transposing:', data.shape)
        assert data.shape[0]==metadata.shape[0]
    return data, metadata

def load_vae(model_path,data,seed=2023,device='cpu'):
    model_here = VAE
    # Hyperparameters:
    idim, md, feat = get_hyperparams(model_path)
    # Importing the model:
    set_seed(seed)
    model_vae = model_here(idim, md, feat)  # Initialize the model
    model_vae.load_state_dict(torch.load(model_path, map_location=torch.device(device)))  # Load the state dictionary
    model_vae.eval()  # Set the model to evaluation mode
    # Apply AE or VAE to all data:
    data_tensor = torch.tensor(data.values).to(torch.float32)
    reconstruction_x, _, _, z, scaler = apply_VAE(data_tensor, model_vae, y=None)
    df_reconstruction_x = pd.DataFrame(reconstruction_x, index=data.index, columns=data.columns)
    df_z = pd.DataFrame(z, index=data.index)

    return df_z, df_reconstruction_x

def apply_rec_net(data,data_vae,network_model_path,recnet_hyperparams_path,seed=2023,device='cpu'):
    idim = data.shape[1]
    # Reconstruction network
    # Hyperparameters:
    hyper = pd.read_csv(recnet_hyperparams_path, index_col=0)
    hyper = [idim] + hyper.values.tolist()[0] + [idim]
    # Importing the model:
    set_seed(seed)
    model_net = model_net_here(hyper)  # Initialize the model
    model_net.load_state_dict(
        torch.load(network_model_path, map_location=torch.device(device)))  # Load the state dictionary
    model_net.eval()  # Set the model to evaluation mode
    rec_tensor = torch.tensor(data_vae.values).to(torch.float32)
    net_output = model_net(rec_tensor)
    df_net_output = pd.DataFrame(net_output.detach().numpy(), index=data.index, columns=data.columns)

    return df_net_output

def get_selected_groups(data, metadata, groups_list):
    selected_metadata = metadata[metadata.isin(groups_list)]
    selected_indices = selected_metadata.index
    selected_samples = data.loc[selected_indices]
    return selected_samples, selected_metadata, selected_indices

def optimal_k_and_clusters(data, save_path,save_fig=True):
    save_silhouete_fig = os.path.join(save_path, 'optimal_k_and_clusters')
    os.makedirs(save_silhouete_fig, exist_ok=True)
    # Range of k values to try
    k_range = range(2, 11)
    n_range = range(2, 12)
    silhouette_scores = []

    for n in tqdm(n_range,desc='Optimal number of clusters'):
        silhouette_scores_n = []
        for k in k_range:
            # Generate the k-neighbors graph
            knn_graph = kneighbors_graph(data, n_neighbors=k, mode='connectivity', include_self=False)

            # Perform Agglomerative Clustering
            model = AgglomerativeClustering(n_clusters=n, connectivity=knn_graph, linkage='ward')
            model.fit(data)

            silhouette_scores_k = silhouette_score(data, model.labels_)
            silhouette_scores_n.append(silhouette_scores_k)
        silhouette_scores.append(silhouette_scores_n)
    df_silhouette_scores = pd.DataFrame(silhouette_scores, index=n_range, columns=k_range)
    # Plot the silhouette scores
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
    for n, ax_y, ax_x in zip(n_range, [0, 1, 2, 3, 4] * 2, [0] * 5 + [1] * 5):
        current_ax = ax[ax_x, ax_y]
        current_ax.plot(k_range, silhouette_scores[n - 2], 'b-d', markersize=8, linewidth=2, label='Silhouette Score',
                        color='cornflowerblue', markerfacecolor='steelblue', markeredgewidth=1.5,
                        markeredgecolor='steelblue')
        current_ax.set_title(f'{n} clusters')
        current_ax.set_xlabel('k')
        current_ax.set_ylabel('Silhouette Score')
        current_ax.set_ylim([-0.05, 0.35])
    plt.tight_layout()  # Adjust layout to prevent overlap
    if save_fig:
        plt.savefig(os.path.join(save_silhouete_fig,'silhouette_scores_latent.png'), dpi=600, bbox_inches='tight',format='png')
        plt.savefig(os.path.join(save_silhouete_fig,'silhouette_scores_latent.svg'), bbox_inches='tight',format='svg')
        plt.savefig(os.path.join(save_silhouete_fig,'silhouette_scores_latent.pdf'), bbox_inches='tight',format='pdf')

    # Get maximum silhouette score
    # max_silhouette_score = np.max(silhouette_scores)
    # Get the index of the maximum silhouette score
    max_silhouette_score_idx = np.unravel_index(np.argmax(silhouette_scores, axis=None),
                                                np.array(silhouette_scores).shape)
    # Get the optimal number of clusters
    optimal_n_clusters = list(n_range)[max_silhouette_score_idx[0]]
    # Get the optimal number of neighbors
    optimal_n_neighbors = list(k_range)[max_silhouette_score_idx[1]]
    print(f'Optimal number of clusters: {optimal_n_clusters}')
    print(f'Optimal number of neighbors: {optimal_n_neighbors}')
    return optimal_n_clusters, optimal_n_neighbors, df_silhouette_scores

def knn_simple(data,metadata,groups_list,n_clusters):
    dict_groups = {key: value for value, key in enumerate(sorted(groups_list))}
    original_labels = metadata[metadata.isin(groups_list)].replace(dict_groups).values
    changed_groups_indices = []
    for k in range(2, 30):
        # Create a k-nearest neighbors graph
        knn_graph = kneighbors_graph(data, n_neighbors=k, mode='connectivity', include_self=False)
        # Perform agglomerative clustering
        model = AgglomerativeClustering(n_clusters=n_clusters, connectivity=knn_graph, linkage='ward')
        model.fit(data)
        new_labels = model.labels_
        changed_groups = np.where(original_labels != new_labels)[0]
        changed_groups_indices.append(changed_groups)
        print(f'k={k}, changed groups:', len(changed_groups), 'percentage:',
              len(changed_groups) / data.shape[0] * 100)
    return changed_groups_indices

def histogram_changed_samples(sorted_counts, cutoff, n_samples_group, group_name, save_path_fig):
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_counts.index, sorted_counts.values, color='cornflowerblue', edgecolor=None)
    # add a vertical line at cutoff position
    plt.axvline(x=cutoff, color='r', linestyle='--')
    # add horizontal lines at the mean and mean+std
    plt.hlines(cutoff, 0, n_samples_group, colors='r', linestyles='--',label='Cut-off')
    plt.xlabel('Patients')
    plt.ylabel('Counts')
    # Change frequency of ticks on x-axis
    # plt.xticks(np.arange(0, 470, step=20),rotation=90,fontsize=8)
    # Remove ticks on x-axis
    plt.xticks([])
    plt.xlim([-1,n_samples_group])
    # plt.ylim([0,600])
    plt.title('Histogram of Elements Sorted by Counts.\n'+group_name)
    plt.legend()
    plt.savefig(os.path.join(save_path_fig, f'histogram_bootstrap_sorted_{group_name}.png'), dpi=600, bbox_inches='tight', format='png')
    plt.savefig(os.path.join(save_path_fig, f'histogram_bootstrap_sorted_{group_name}.svg'), bbox_inches='tight', format='svg')
    plt.savefig(os.path.join(save_path_fig, f'histogram_bootstrap_sorted_{group_name}.pdf'), bbox_inches='tight', format='pdf')

def knn_bootstrapping(data,data_selected, metadata,metadata_selected,n_neighbors, n_clusters,p=0.05,n_samples=100, n_boot=1000, replacement=False, save_path=None):
    if save_path is not None:
        save_path_fig = os.path.join(save_path, 'knn_bootstrapping')
        os.makedirs(save_path_fig, exist_ok=True)
    # Get unique groups
    groups_list = metadata_selected.unique()
    # Count number of samples in each group
    samples_dict = {group: metadata_selected[metadata_selected == group].index for group in groups_list}
    samples_per_group = {group: metadata_selected.value_counts()[group] for group in groups_list}
    print('Number of samples in each group:', samples_per_group)
    # Probability of each group to change during bootstrapping:
    # I.e., the probability that at each bootstrapping iteration, a sample from a group changes group.
    # That is, the number of samples in each group divided by the total number of samples.
    # We simulate a coin toss by multiplying this probability by 0.5, the probability of changing group.
    # The total probability implies multiplying by the number of bootstraps.
    prob_group = {group: n_boot*0.5*n_samples/samples_per_group[group] for group in groups_list}
    # Dictionary of groups
    dict_groups = {key: value for value, key in enumerate(sorted(groups_list))}
    # Get the indices of the groups,
    indices_groups = [metadata_selected[metadata_selected == group].index for group in groups_list]
    # Select df to perform kNN
    original_labels = metadata_selected.replace(dict_groups)
    changed_groups_patients = []
    n_samples_list = []
    dict_clusters = {key: [] for key in original_labels.index}
    # Repeat bootstrapping n_boot times
    for i in tqdm(range(n_boot),desc='Bootstrapping'):
        n_samples_list.append(n_samples)
        # Bootstrap samples
        X_sample = [
            resample(data_selected.loc[indices], replace=replacement, n_samples=n_samples)
            for indices in indices_groups
        ]
        X_sample = pd.concat(X_sample)

        # Get sampled patients and labels:
        sampled_patients = X_sample.index
        sampled_labels = original_labels.loc[sampled_patients].values
        # Get patients that have not been sampled
        non_sampled_patients = original_labels.index.difference(sampled_patients)
        none_labels = [None] * len(non_sampled_patients)
        # Get graph:
        knn_graph = kneighbors_graph(X_sample, n_neighbors=n_neighbors, mode='connectivity', include_self=False)

        # Agglomerative clustering on knn_graph
        model_agg = AgglomerativeClustering(n_clusters=n_clusters, connectivity=knn_graph, linkage='ward')
        model_agg.fit(X_sample)

        # Get new labels
        new_labels = model_agg.labels_
        # Add the new labels to the dictionary
        for patient, label in zip(sampled_patients.tolist() + non_sampled_patients.tolist(),
                                  new_labels.tolist() + none_labels):
            dict_clusters[patient].append(label)
        # Check if the labels have changed
        changed_labels = np.where(sampled_labels != new_labels)[0]
        changed_patients_boot = sampled_patients[changed_labels]
        changed_groups_patients.append(changed_patients_boot)
    df_clusters = pd.DataFrame(dict_clusters)
    avg_change = np.mean([len(i) for i in changed_groups_patients])
    print('Average number of changed labels:', avg_change)
    print('As a percentage:', avg_change / n_samples * 100)

    # flatten changed_groups_indices
    flatten_changed_groups_patients = []
    for i in changed_groups_patients:
        flatten_changed_groups_patients += i.tolist()
    # Get counts of each element
    bootstrap_counts = pd.Series(flatten_changed_groups_patients).value_counts()
    bootstrap_counts = pd.DataFrame(bootstrap_counts)
    bootstrap_counts['group'] = metadata_selected.loc[bootstrap_counts.index]
    print('bootstrap_counts.head():', bootstrap_counts.head())
    print('bootstrap_counts group value_counts:', bootstrap_counts['group'].value_counts())
    print('bootstrap_counts counts column head:', bootstrap_counts['count'].head())
    # Print min, max and mean counts per group
    for group_i in groups_list:
        print(f'Min count for {group_i}:', bootstrap_counts['count'] [bootstrap_counts['group'] == group_i].min())
        print(f'Max count for {group_i}:', bootstrap_counts['count'][bootstrap_counts['group'] == group_i].max())
        print(f'Mean count for {group_i}:', bootstrap_counts['count'][bootstrap_counts['group'] == group_i].mean())
    # wait 30 seconds
    # import time
    # time.sleep(30)
    # Get the patients that changes more than average + std
    # n = (bootstrap_counts > np.mean(bootstrap_counts) + np.std(bootstrap_counts)).sum()
    # Get the top 5% of the patients that changed the most
    # n = int(p * data_selected.shape[0])
    # Get the patients that changes more than expected by chance: count_group_i > prob_group_i
    n = {group: prob_group[group] for group in groups_list}
    print('n =', n)
    # bootstrap_most_changing_patients = bootstrap_counts.sort_values(ascending=False).head(n).index
    bootstrap_most_changing_patients = {
        group: bootstrap_counts[(bootstrap_counts['group'] == group) & (bootstrap_counts['count'] > n[group])].index
        for group in groups_list
    }
    print('bootstrap_most_changing_patients:', bootstrap_most_changing_patients)
    print('length of bootstrap_most_changing_patients:', {group: len(bootstrap_most_changing_patients[group]) for group in groups_list})
    # Count the occurrences of each element
    counts = pd.Series(flatten_changed_groups_patients).value_counts()
    # Sort the elements based on their counts
    sorted_counts = counts.sort_values(ascending=False)
    sorted_counts.name = 'count'
    sorted_counts = pd.DataFrame(sorted_counts)
    sorted_counts['group'] = metadata_selected.loc[sorted_counts.index]
    print('sorted_counts.head():', sorted_counts.head())
    metadata_bootstrap = metadata.copy()
    print('metadata.value_counts():', metadata.value_counts())
    for group_i in groups_list:
        metadata_bootstrap.loc[bootstrap_most_changing_patients[group_i]] = 'G3-G4'
    print('metadata_bootstrap.value_counts():', metadata_bootstrap.value_counts())
    contingency_table = pd.crosstab(metadata, metadata_bootstrap, margins=True)
    print('Contingency table:', contingency_table)
    # Plot histogram of the number of changes for each sample and group
    for group_i in groups_list:
        changed_samples = sorted_counts['count'][sorted_counts['group']==group_i]
        n_samples_group = changed_samples.shape[0]
        print('n_samples_group:', n_samples_group)
        histogram_changed_samples(
            changed_samples,
            cutoff=n[group_i],
            n_samples_group=n_samples_group,
            group_name=group_i,
            save_path_fig=save_path_fig)
    # UMAP of detected patients
    plot_umap(data.T, metadata_bootstrap, dict_umap,
              n_components=2, save_fig=True,
              save_as=os.path.join(save_path_fig,'umap_bootstrap'),
              seed=2023,
              title=None)
    return df_clusters,metadata_bootstrap,contingency_table, sorted_counts



def main(args):
    groups_list = ['Group 3', 'Group 4']
    #groups_list = ['Group 3', 'Group 4','Synthetic','G3-G4']
    data, metadata = load_data(args.data_path, args.metadata_path)
    # df_z, df_reconstruction_x = load_vae(args.model_path, data)
    # df_net_output = apply_rec_net(data, df_reconstruction_x,network_model_path=args.network_model_path,recnet_hyperparams_path=args.recnet_hyperparams_path)
    # Get only G3 and G4 patients' rnaseq data:
    data_selected, metadata_selected, indices_selected = get_selected_groups(data, metadata, groups_list)
    # df_z_selected, _, _ = get_selected_groups(df_z, metadata, groups_list)
    # df_net_output_selected, _, _ = get_selected_groups(df_net_output, metadata, groups_list)
    # Optimal number of clusters
    print("Shape of df passed to optimal_k_and_clusters:", data_selected.shape)
    optimal_n_clusters, optimal_n_neighbors, df_silhouette_scores = optimal_k_and_clusters(data_selected, save_path)
    df_silhouette_scores.to_csv(os.path.join(save_path, 'silhouette_scores.csv'))
    # kNN bootstrapping
    df_clusters,metadata_bootstrap,contingency,sorted_counts = knn_bootstrapping(
        # df_z, df_z_selected, metadata, metadata_selected, # old version: in-code latent space
        data, data_selected, metadata, metadata_selected, # new version: use imported data
        optimal_n_neighbors, optimal_n_clusters,
        n_samples=100, n_boot=1000,
        replacement=False, save_path=save_path)

    # Save the results
    df_clusters.to_csv(os.path.join(save_path, 'bootstrap_clusters.csv'))
    metadata_bootstrap.to_csv(os.path.join(save_path, 'metadata_after_bootstrap.csv'))
    contingency.to_csv(os.path.join(save_path, 'contingency_table.csv'))
    contingency.to_latex(os.path.join(save_path, 'contingency_table.tex'))
    sorted_counts.to_csv(os.path.join(save_path, 'sorted_counts.csv'))



if __name__ == '__main__':
    import argparse

    # Degine arguments
    parser = argparse.ArgumentParser(description='Clustering for G3 and G4')
    parser.add_argument('--data_path', type=str,
                        default='data/interim/20240301_Mahalanobis/cavalli.csv',
                        help='Path to the data')
    parser.add_argument('--metadata_path', type=str,
                        default='data/cavalli_subgroups.csv',
                        help='Path to the metadata')
    # parser.add_argument('--model_path', type=str,
    #                     default='models/20240417_cavalli_maha/20240417_VAE_idim12490_md2048_feat16mse_relu.pth',
    #                     help='Path to the model')
    # parser.add_argument('--network_model_path', type=str,
    #                     default="data/interim/20240802_adjust_reconstruction/network_reconstruction.pth",
    #                     help='Path to the reconstruction network model')
    # parser.add_argument('--recnet_hyperparams_path', type=str,
    #                     help='Path to the reconstruction network hyperparameters')
    parser.add_argument('--save_path', type=str,
                        default=None,
                        help='Path to save the results')

    # Parse arguments
    args = parser.parse_args()

    # Save path
    if args.save_path is None:
        from datetime import datetime

        today = datetime.today().strftime('%Y%m%d')
        save_path = f'data/interim/{today}_clustering_g3g4'
    else:
        save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    # Dictionary of colors for the different groups to plot UMAP
    global dict_umap
    dict_umap = {'SHH': '#b22222', 'WNT': '#6495ed', 'Group 3': '#ffd700', 'Group 4': '#008000', 'G3-G4': '#db7093'}
    main(args)

# Running with augmented data:
# python clustering_g3g4.py --data_path data/interim/20240801_data_augmentation/rnaseq_synth.csv --metadata_path data/interim/20240801_data_augmentation/clinical_synth.csv  --save_path data/interim/20240801_data_augmentation/clustering_g3g4