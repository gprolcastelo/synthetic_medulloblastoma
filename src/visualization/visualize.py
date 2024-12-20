# UMAP: choose  2 or 3 components
import os
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import umap
from sklearn.manifold import TSNE

def str_or_list(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(',')]
    elif isinstance(value, list):
        return value
    else:
        raise argparse.ArgumentTypeError("Value must be a string or a list")

def plot_umap(data, clinical, colors_dict, n_components=2, save_fig=False, save_as=None, seed=None, title='UMAP',show=True):
    '''
    Plot UMAP of the data with different colors for the different groups.
    :param data:
    :param clinical:
    :param colors_dict:
    :param n_components:
    :param save_fig:
    :param save_as:
    :param seed:
    :return:
    '''
    # Check number of samples is the second dimension of data:
    if data.shape[1] != clinical.shape[0]:
        data = data.T
    # Check number of samples coincides with metadata:
    assert data.shape[1] == clinical.shape[0], "Data and metadata have different number of samples"
    # # Check all samples are in the metadata:
    # assert (data.index.isin(clinical.index)).all(), "Data and metadata contain different samples"
    if seed is not None:
        umap_ = umap.UMAP(n_components=n_components, random_state=seed)
    else:
        umap_ = umap.UMAP(n_components=n_components)
    X_umap = umap_.fit_transform(data.T)
    # Plot UMAP
    plt.figure(figsize=(10, 10))
    # Use different colors for the different groups: SHH, WNT, Group 3, Group 4
    # colors = {'SHH': 'tomato', 'WNT': 'cornflowerblue', 'Group 3': 'limegreen', 'Group 4': 'aqua'}
    # Get the list of all patient IDs
    all_patients = data.columns.tolist()
    if n_components == 2:
        for group, color in colors_dict.items():
            # Get the patient IDs for the current group
            group_patients = clinical[clinical == group].index.tolist()
            # Convert the patient IDs to integer indices
            idx = [all_patients.index(patient) for patient in group_patients]
            plt.scatter(X_umap[idx, 0], X_umap[idx, 1], c=color, label=group)
        plt.legend(fontsize=16)
        plt.title(title, fontsize=20)
        plt.xlabel('UMAP 1', fontsize=16)
        plt.ylabel('UMAP 2', fontsize=16)
        plt.xticks([])
        plt.yticks([])
        # Change the color of the axis lines to black
        plt.gca().spines['top'].set_color('black')
        plt.gca().spines['right'].set_color('black')
        plt.gca().spines['left'].set_color('black')
        plt.gca().spines['bottom'].set_color('black')
    else:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        for group, color in colors_dict.items():
            # Get the patient IDs for the current group
            group_patients = clinical[clinical == group].index.tolist()
            # Convert the patient IDs to integer indices
            idx = [all_patients.index(patient) for patient in group_patients]
            ax.scatter(X_umap[idx, 0], X_umap[idx, 1], X_umap[idx, 2], c=color, label=group)
        ax.legend(fontsize=16)
        ax.set_title(title)
        ax.set_xlabel('UMAP 1', fontsize=16)
        ax.set_ylabel('UMAP 2', fontsize=16)
        ax.set_zlabel('UMAP 3', fontsize=16)
        # Change the color of the axis lines to black
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        # Save to path_figs
        # save_as = os.path.join(path_figs, f"{today}_3D_UMAP")
        # Remove all axe ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    # Set the background color to white
    plt.gca().set_facecolor('white')
    # Save figure as png and svg:
    if save_fig:
        plt.savefig(f'{save_as}.png', format='png', bbox_inches='tight', dpi=1000, transparent=True)
        plt.savefig(f'{save_as}.svg', format='svg', bbox_inches='tight')
        plt.savefig(f'{save_as}.pdf', format='pdf', bbox_inches='tight')
    # Display the plot
    plt.tight_layout()
    if show:
        plt.show()



def plot_tsne(data, clinical, colors_dict, n_components=2, save_fig=False, save_as=None, seed=None, title='t-SNE', show=True):
    '''
    Plot t-SNE of the data with different colors for the different groups.
    :param data: DataFrame, the data to be plotted
    :param clinical: Series, the clinical labels
    :param colors_dict: dict, mapping of clinical groups to colors
    :param n_components: int, number of components for t-SNE
    :param save_fig: bool, whether to save the figure
    :param save_as: str, path to save the figure
    :param seed: int, random seed for reproducibility
    :param title: str, title of the plot
    :param show: bool, whether to show the plot
    :return: None
    '''
    today = datetime.now().strftime("%Y%m%d")

    if seed is not None:
        tsne = TSNE(n_components=n_components, random_state=seed)
    else:
        tsne = TSNE(n_components=n_components)
    X_tsne = tsne.fit_transform(data.T)

    plt.figure(figsize=(10, 10))
    all_patients = data.columns.tolist()
    if n_components == 2:
        for group, color in colors_dict.items():
            group_patients = clinical[clinical == group].index.tolist()
            idx = [all_patients.index(patient) for patient in group_patients]
            plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=color, label=group)
        plt.legend(fontsize=16)
        plt.title(title)
        plt.xlabel('t-SNE 1', fontsize=16)
        plt.ylabel('t-SNE 2', fontsize=16)
    else:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        for group, color in colors_dict.items():
            group_patients = clinical[clinical == group].index.tolist()
            idx = [all_patients.index(patient) for patient in group_patients]
            ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], X_tsne[idx, 2], c=color, label=group)
        ax.legend(fontsize=16)
        ax.set_title(title)
        ax.set_xlabel('t-SNE 1', fontsize=16)
        ax.set_ylabel('t-SNE 2', fontsize=16)
        ax.set_zlabel('t-SNE 3', fontsize=16)

    if save_fig:
        plt.savefig(f'{save_as}.png', format='png', bbox_inches='tight', dpi=1000, transparent=True)
        plt.savefig(f'{save_as}.svg', format='svg', bbox_inches='tight')
        plt.savefig(f'{save_as}.pdf', format='pdf', bbox_inches='tight')
    plt.tight_layout()
    if show:
        plt.show()

if __name__=='__main__':
    import pandas as pd
    import argparse
    parser = argparse.ArgumentParser(description='Visualize data using UMAP')
    parser.add_argument('--data_path', type=str, help='Path to the data file')
    parser.add_argument('--metadata_path', type=str, help='Path to the clinical file')
    parser.add_argument('--save_path', type=str, help='Path to save the figure')
    parser.add_argument('--groups', type=str_or_list, help='Names of groups in the data')
    parser.add_argument('--n_components', type=int, help='Number of components for UMAP or t-SNE', default=2)
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--title', type=str, help='Title of the UMAP', default='umap')

    args = parser.parse_args()
    groups = args.groups
    if 'G3-G4' in groups:
        groups = ['G3-G4' if x == 'G3-G4' else x for x in groups]
    print('groups:', groups)
    os.makedirs(args.save_path, exist_ok=True)
    # Load the data and metadata
    data = pd.read_csv(args.data_path, index_col=0)
    clinical = pd.read_csv(args.metadata_path, index_col=0)
    clinical.replace({'Group3': 'Group 3', 'Group4': 'Group 4'},inplace=True) # Just in case, adjust the names
    clinical = clinical.squeeze()
    print(type(clinical))
    print('clinical.value_counts():', clinical.value_counts())
    dict_plot = {'SHH': '#b22222',
                 'WNT': '#6495ed',
                 'Group 3': '#ffd700',
                 'Group 4': '#008000',
                 'G3-G4': '#db7093',
                 'Synthetic': '#808080'}
    # Select the groups to plot
    clinical = clinical[clinical.isin(groups)]
    try:
        data = data.loc[clinical.index]
    except KeyError:
        try:
            data = data[clinical.index]
        except KeyError:
            raise ValueError("No patients found in data")
    dict_plot = {k: v for k, v in dict_plot.items() if k in groups}
    print('clinical.value_counts():', clinical.value_counts())
    # Replace literal \n with an actual newline character
    title = args.title.replace('\\n', '\n')
    # Plot UMAP
    plot_umap(
        data=data,
        clinical=clinical,
        colors_dict=dict_plot,
        n_components=args.n_components,
        save_fig=True,
        save_as=os.path.join(args.save_path,'umap'),
        seed=args.seed,
        show=False,
        title=title)

# Example usage from the command line:
# python src/visualization/visualize.py --data_path data/interim/20241023_preprocessing/rnaseq_maha.csv \
#                                       --metadata_path data/cavalli_subgroups.csv \
#                                       --save_path figures/umap \
#                                       --groups SHH WNT Group3 Group4 \
#                                       --n_components 2 \
#                                       --seed 2023