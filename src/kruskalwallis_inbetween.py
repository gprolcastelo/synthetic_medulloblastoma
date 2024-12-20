import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from scipy.stats import kruskal
import scikit_posthocs as sp

plt.style.use('ggplot')


def str_or_list(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(',')]
    elif isinstance(value, list):
        return value
    else:
        raise argparse.ArgumentTypeError("Value must be a string or a list")



def load_data(path_data, path_clinical, path_genes, list_groups):
    data = pd.read_csv(path_data, index_col=0)
    clinical = pd.read_csv(path_clinical, index_col=0).replace({'Group3': 'Group 3', 'Group4': 'Group 4'})
    clinical = pd.Series(clinical.values.flatten(), index=clinical.index, name='group')  # Convert to Series
    if path_genes == 'all':
        genes = data.columns
    else:
        genes = pd.read_csv(path_genes, index_col=0)
        genes = pd.Series(genes.values.flatten(), index=genes.index, name='genes')  # Convert to Series
        # Select only genes of interest
    try:
        data_for_clustering = data.loc[genes]
    except KeyError:
        try:
            data_for_clustering = data[genes]
        except KeyError:
            raise ValueError("No genes found in data")
    # Select only patients with a group label
    clinical = clinical.loc[clinical.isin(list_groups)]
    print('clinical selected groups:', clinical.value_counts())
    try:
        data_for_clustering = data_for_clustering[clinical.index]
    except KeyError:
        try:
            data_for_clustering = data_for_clustering.loc[clinical.index]
        except KeyError:
            raise ValueError("No patients found in data")
    return data_for_clustering, clinical, genes



def apply_kw_nonpairwise(data, clinical):
    # Take into account possible name variations of the groups
    clinical.replace(
        {'Group3': 'Group 3', 'synthetic_Group 3': 'Group 3',
         'Group4': 'Group 4', 'synthetic_Group 4': 'Group 4',
         'G3-G4': 'G3-G4', 'synthetic_G3-G4': 'G3-G4'
         },
        inplace=True)
    print('inside kw, clinical:', clinical.value_counts())
    # Kruskal-Wallis test:
    # - Null hypothesis: the population median of compared groups are equal (p-value > 0.05)
    # - Alternative hypothesis: the population median of at least one group is different from the others (p-value < 0.05)
    # Create a DataFrame to store the p-values of the Kruskal-Wallis test

    # columns_here = ['g3_g4', 'g3_transition', 'g4_transition']
    columns_here = ['g3_g4_transition']
    p_values_df = pd.DataFrame(index=data.columns, columns=columns_here)
    h_values_df = pd.DataFrame(index=data.columns, columns=columns_here)

    # Create a dictionary mapping column names to the datasets to be compared
    comparison_dict = {
        'g3_g4_transition': (
            data.loc[clinical[clinical == 'Group 3'].index],
            data.loc[clinical[clinical == 'Group 4'].index],
            data.loc[clinical[clinical == 'G3-G4'].index])
    }

    # Iterate over the genes (rows) in the datasets
    for gene in tqdm(data.columns, desc='Applying KW non-pairwise'):
        for col_name, (df_i, df_j, df_k) in comparison_dict.items():
            # Check if all values are identical in all three groups
            if np.all(
                    df_i[gene] == df_i[gene].iloc[0]
            ) and np.all(
                df_j[gene] == df_j[gene].iloc[0]
            ) and np.all(
                df_k[gene] == df_k[gene].iloc[0]
            ):
                p_value = 1.0  # Assign 1 to the p-value if all values are identical
                h = 0.0
            else:
                # Perform the Kruskal-Wallis test
                h, p_value = kruskal(df_i[gene], df_j[gene], df_k[gene])

            # Store the p-value and h-value
            p_values_df.loc[gene, col_name] = p_value
            h_values_df.loc[gene, col_name] = h

    return p_values_df, h_values_df


def apply_dunn(data, clinical):
    # Take into account possible name variations of the groups
    clinical.replace(
        {'Group3': 'Group 3', 'synthetic_Group 3': 'Group 3',
         'Group4': 'Group 4', 'synthetic_Group 4': 'Group 4',
         'G3-G4': 'G3-G4', 'synthetic_G3-G4': 'G3-G4'
         },
        inplace=True)
    # print('inside dunn, clinical:', clinical.value_counts())

    # Dunn's post-hoc test:
    # - Null hypothesis: the population median of compared groups are equal (p-value > 0.05)
    # - Alternative hypothesis: the population median of compared groups are different (p-value < 0.05)

    columns_here = ['g3_g4', 'g3_transition', 'g4_transition']
    p_values_df = pd.DataFrame(index=data.columns, columns=columns_here)
    # print('p_values_df.shape:', p_values_df.shape)

    # Create a dictionary mapping column names to the datasets to be compared
    comparison_dict = {
        'g3_g4': (data.loc[clinical[clinical == 'Group 3'].index],
                  data.loc[clinical[clinical == 'Group 4'].index],
                  data.loc[clinical[clinical == 'G3-G4'].index])
    }

    # Iterate over the genes (rows) in the datasets
    for gene in tqdm(data.columns, desc='Applying Dunn to groups 3, 4, and G3-G4'):
        # print('gene:', gene)
        for col_name, (df_g3, df_g4, df_transition) in comparison_dict.items():
            # print('col_name:', col_name)
            # Check if all values are identical in all groups
            if np.all(
                    df_g3[gene] == df_g3[gene].iloc[0]
            ) and np.all(
                df_g4[gene] == df_g4[gene].iloc[0]
            ) and np.all(
                df_transition[gene] == df_transition[gene].iloc[0]
            ):
                p_value = 1.0  # Assign 1 to the p-value if all values are identical
            else:
                # Perform Dunn's test
                p_value = sp.posthoc_dunn([df_g3[gene], df_g4[gene], df_transition[gene]], p_adjust='bonferroni')
                # print('p_value:', p_value)
            # Store the p-value
            p_value.columns = ['Group 3', 'Group 4', 'G3-G4']
            p_value.index = ['Group 3', 'Group 4', 'G3-G4']

            # Store the p-values in the DataFrame
            p_values_df.loc[gene, 'g3_g4'] = p_value.at['Group 3', 'Group 4']
            p_values_df.loc[gene, 'g3_transition'] = p_value.at['Group 3', 'G3-G4']
            p_values_df.loc[gene, 'g4_transition'] = p_value.at['Group 4', 'G3-G4']

    return p_values_df


def plot_differential_genes(data, clinical, genes, p_values_df, path_boxplot):
    # Define colors for the boxplots
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    # Define genes to be plotted
    # genes = ['MYC', 'SNCAIP']
    # genes = ['ABCC8', 'ABHD2', 'ABI3BP', 'ABLIM1']
    # Labels for each subplot
    # subplot_labels = ['a)', 'b)']
    # Create a 2x2 subplot

    # # Flatten the axes array for easy iteration
    # axs_flat = axs.flatten()

    # Iterate over genes and corresponding axes
    for gene in tqdm(genes, desc='Creating boxplots'):
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        ax = axs
        # Get the datasets to be compared
        df_g3 = data[clinical[clinical == 'Group 3'].index].loc[gene]
        df_g4 = data[clinical[clinical == 'Group 4'].index].loc[gene]
        df_transition = data[clinical[clinical == 'G3-G4'].index].loc[gene]
        # Concatenate the datasets
        df_gene = [df_g3, df_transition, df_g4]
        # Plot the boxplot on the corresponding subplot
        bp = ax.boxplot(df_gene, patch_artist=True, labels=['Group 3', 'G3-G4', 'Group 4'])
        # Set colors for each box
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        for median in bp['medians']:
            median.set(color='navy', linewidth=5)
        ax.set_title(gene, fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_ylabel('Gene Expression', fontsize=8)
        # ax.set_ylim(0, 20)
        # Perform the Mann-Whitney U test between Group 3 and Group 4
        # stat, p_value = mannwhitneyu(df_g3, df_g4)
        p_value_g3g4 = p_values_df.loc[gene, 'g3_g4']
        p_value_g3transition = p_values_df.loc[gene, 'g3_transition']
        p_value_g4transition = p_values_df.loc[gene, 'g4_transition']
        # Add a line between the two boxplots
        x1, x2, x3 = 1, 2, 3
        y, col = max(max(df_g3), max(df_g4)) + 2, 'gray'
        h1, h2, h3 = 1, .5, 0
        ax.plot([x1, x1, x3, x3], [y + .75, y + h1, y + h1, y + .75], lw=1, c=col)
        ax.plot([x1, x1, x2, x2], [y + .25, y + h2, y + h2, y + .25], lw=1, c=col)
        ax.plot([x2, x2, x3, x3], [y - .25, y + h3, y + h3, y - .25], lw=1, c=col)

        # Annotate with asterisks based on the p-value
        if p_value_g3g4 < 0.001:
            text_1 = '***'
        elif p_value_g3g4 < 0.01:
            text_1 = '**'
        elif p_value_g3g4 < 0.05:
            text_1 = '*'
        else:
            text_1 = 'ns'  # not significant

        if p_value_g3transition < 0.001:
            text_2 = '***'
        elif p_value_g3transition < 0.01:
            text_2 = '**'
        elif p_value_g3transition < 0.05:
            text_2 = '*'
        else:
            text_2 = 'ns'

        if p_value_g4transition < 0.001:
            text_3 = '***'
        elif p_value_g4transition < 0.01:
            text_3 = '**'
        elif p_value_g4transition < 0.05:
            text_3 = '*'
        else:
            text_3 = 'ns'

        ax.text((x1 + x2) * .667, y + h1, text_1, ha='center', va='bottom', color=col)
        ax.text((x1 + x3) * .375, y + h2, text_2, ha='center', va='bottom', color=col)
        ax.text((x2 + x3) * .5, y + h3, text_3, ha='center', va='bottom', color=col)
        # Add subplot label
        # ax.text(-0.1, 1.1, subplot_labels[i], transform=ax.transAxes, fontsize=32, fontweight='bold', va='top', ha='right')
        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the figure
        plt.savefig(os.path.join(path_boxplot, f'{gene}_boxplot.png'), dpi=600, bbox_inches='tight', format='png')
        plt.savefig(os.path.join(path_boxplot, f'{gene}_boxplot.svg'), bbox_inches='tight', format='svg')
        plt.savefig(os.path.join(path_boxplot, f'{gene}_boxplot.pdf'), bbox_inches='tight', format='pdf')
        # Close and clear the figure
        plt.close(); plt.clf(); fig.clf()
    # plt.show()


# %%
def plot_differential_genes_to_pdf_doc(data, clinical, genes, path_boxplot, nrows=6, ncols=4):
    # Sort genes alphabetically
    genes = sorted(genes)
    # Define colors for the boxplots
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    # A4 dimensions in inches (8.27 x 11.69)
    a4_width, a4_height = 8.27, 11.69
    # Create a PdfPages object
    pdf_path = os.path.join(path_boxplot, '0_differential_genes_boxplots.pdf')
    print()
    with PdfPages(pdf_path) as pdf:
        # Iterate over genes and corresponding axes
        for i, gene in enumerate(tqdm(genes, desc='Creating boxplots')):
            if i % (nrows * ncols) == 0:
                if i > 0:
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(a4_width, a4_height))
                axs_flat = axs.flatten()
                # Add margins
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)

            ax = axs_flat[i % (nrows * ncols)]
            # Get the datasets to be compared
            df_g3 = data.loc[clinical[clinical == 'Group 3'].index][gene]
            df_g4 = data.loc[clinical[clinical == 'Group 4'].index][gene]
            df_transition = data.loc[clinical[clinical == 'G3-G4'].index][gene]
            # Concatenate the datasets
            df_gene = [df_g3, df_transition, df_g4]
            # Plot the boxplot on the corresponding subplot
            bp = ax.boxplot(df_gene, patch_artist=True, labels=['Group 3', 'G3-G4', 'Group 4'])
            # Set colors for each box
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            for median in bp['medians']:
                median.set(color='navy', linewidth=2)
            ax.set_title(gene, fontsize=8)
            ax.tick_params(axis='x', labelsize=5)
            ax.tick_params(axis='y', labelsize=5)
            # ax.set_ylim(0, 20)

        # Save the last page
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main(args):
    os.makedirs(args.path_boxplot, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    # Load data:
    # list_groups = ['Group 3','Group 4','G3-G4']
    # list_groups = ['synthetic_Group 3','synthetic_Group 4','synthetic_G3-G4'] # choose synthetic groups
    list_groups = args.group_to_analyze
    print(f'Groups to augment: {list_groups}')
    data_for_clustering, clinical, genes = load_data(
        path_data=args.path_data,
        path_clinical=args.path_clinical,
        path_genes=args.path_genes,
        list_groups=list_groups)

    # Perform Kruskal-Wallis test:
    p_values_kw, h_values_kw = apply_kw_nonpairwise(data_for_clustering, clinical)
    p_values_kw.to_csv(os.path.join(args.save_path, 'p_values_kw.csv'))
    p_values_kw.to_csv(os.path.join(args.save_path, 'h_values_kw.csv'))

    # Get genes that are differentially expressed in Kruskal-Wallis test:
    p_values_filtered = p_values_kw[p_values_kw['g3_g4_transition'] < args.alpha]
    genes_filtered = p_values_filtered.index
    print('Number of genes after filtering significant differences after KW:', genes_filtered.shape[0])

    # Perform Dunn's post-hoc test on filtered genes:
    p_values_dunn = apply_dunn(data_for_clustering[genes_filtered], clinical)
    p_values_dunn.to_csv(os.path.join(args.save_path, 'p_values_dunn.csv'))
    # Filter genes with all p-values < alpha:
    p_values_dunn_filtered = p_values_dunn[
        (p_values_dunn['g3_g4'] < args.alpha) & (p_values_dunn['g3_transition'] < args.alpha) & (
                    p_values_dunn['g4_transition'] < args.alpha)]
    always_diff_genes = p_values_dunn_filtered.index
    always_diff_genes = pd.Series(always_diff_genes, name='genes')
    print('Number of genes showing significant differences in all groups after Dunn:', always_diff_genes.shape[0])
    always_diff_genes.to_csv(os.path.join(args.save_path, 'always_diff_genes.csv'))

    # plot_differential_genes(data_for_clustering.T,clinical,genes=p_values_dunn_filtered.index,p_values_df=p_values_dunn_filtered,path_boxplot=args.path_boxplot)

    plot_differential_genes_to_pdf_doc(data=data_for_clustering, clinical=clinical, genes=p_values_dunn_filtered.index, path_boxplot=args.path_boxplot, )

    print('Done!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Example argparse in Jupyter notebook')
    parser.add_argument('--path_data', type=str, help='Path to RNA-seq data')
    parser.add_argument('--path_clinical', type=str, help='Path to clinical data')
    parser.add_argument('--path_genes', type=str, help='Path to genes data', default='all')
    parser.add_argument('--alpha', type=float, default=0.01, help='Significance level for the Kruskal-Wallis test')
    parser.add_argument('--path_boxplot', type=str, help='Path to save boxplot figures')
    parser.add_argument('--save_path', type=str, help='Path to save the results')
    parser.add_argument('--group_to_analyze', type=str_or_list, default='G3-G4', help='Group to augment')
    # Simulate command-line arguments
    # import sys
    #
    # sys.argv = ['notebook',  # notebook simulates the script name, then the arguments
    #             '--path_data', 'data/interim/20241115_data_augmentation/real/augmented_data.csv',
    #             '--path_clinical', 'data/interim/20241115_data_augmentation/real/augmented_clinical.csv',
    #             '--path_genes', 'all',
    #             '--alpha', '0.01',
    #             '--path_boxplot', 'reports/figures/20241122_kw/boxplot_augmented/synth_patients',
    #             '--save_path', 'data/processed/20241122_differentially_expressed_genes/synth_patients',
    #             '--group_to_analyze', 'synthetic_Group 3, synthetic_Group 4, synthetic_G3-G4'
    #             ]
    args = parser.parse_args()
    print('args =', args)


    main(args)


# Use with:
# Augmented data
# python src/kruskalwallis_inbetween.py --path_data data/interim/20241115_data_augmentation/real/augmented_data.csv \
#                                       --path_clinical data/interim/20241115_data_augmentation/real/augmented_clinical.csv \
#                                       --path_genes 'all' \
#                                       --alpha 0.01 \
#                                       --path_boxplot reports/figures/20241115_kw/boxplot_augmented/synth_patients \
#                                       --save_path data/processed/20241115_differentially_expressed_genes/synth_patients \
#                                       --group_to_analyze 'synthetic_Group 3, synthetic_Group 4, synthetic_G3-G4'