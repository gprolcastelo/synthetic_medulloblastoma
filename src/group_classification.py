import os
os.environ['NUMEXPR_MAX_THREADS'] = '112'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
plt.style.use('ggplot')
# Custom functions:
from src.data_processing.classification import *

def boxplot_metrics(plot_results,legend_labels,n_seeds,save_dir,cmap='Pastel2',title=''):
    """
    Boxplot of the metrics of the classification
    :param plot_results: list of classification results
    :param legend_labels: list of labels for the metrics
    :param save_dir: path to save the plot
    :param cmap: colormap to use for the plot (changes box fill color)
    """
    metrics_list = np.array([plot_results[i] for i in range(n_seeds)])
    # mean_metrics = np.mean(metrics_list,axis=0)
    # std_metrics = np.std(metrics_list,axis=0)
    # legend_labels = metrics_mean.columns
    # cmap = 'Pastel2'
    colors = colormaps[cmap](np.linspace(0, 1, 7))
    plt.figure(figsize=(9, 7))
    bp=plt.boxplot(metrics_list[:,0], labels=legend_labels, patch_artist=True)
    # Rotate the x-axis labels
    plt.xticks(rotation=45,fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0,1.05])
    plt.xlabel('Score', fontsize=16)
    plt.title(title, fontsize=18)
    # Set the colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set(color='navy', linewidth=2)
    plt.savefig(os.path.join(save_dir,'metrics_boxplot.png'),bbox_inches='tight',dpi=600,format='png')
    plt.savefig(os.path.join(save_dir,'metrics_boxplot.svg'),bbox_inches='tight',format='svg')
    plt.savefig(os.path.join(save_dir,'metrics_boxplot.pdf'),bbox_inches='tight',format='pdf')
    #plt.show()

def load_data(path_data,path_metadata):
    # Microarray data:
    data = pd.read_csv(path_data, index_col=0)
    # Load the subgroup labels:
    metadata = pd.read_csv(path_metadata, index_col=0).squeeze()
    # Make sure first dimension of both data and metadata is the same
    if data.shape[0] != metadata.shape[0]:
        data = data.T
    assert data.shape[0] == metadata.shape[0] , 'Data and metadata have different number of samples'
    return data, metadata



#################
# Classification
#################
def main(path_data,path_metadata,n_classes,n_seeds=10,n_trials_optuna=100,n_br=100,classification_type='weighted',test_size=0.2,n_threads=16,save_dir='.'):
    # Load data
    data, labels = load_data(path_data=path_data, path_metadata=path_metadata)
    os.makedirs(save_dir, exist_ok=True)
    # Seeds for classification
    seeds = np.random.randint(0,1e9,n_seeds)
    # Run through the seeds and perform classification
    metrics = []
    for seed_i in tqdm(seeds):
        classification = classification_benchmark(
            X_data=data,
           y_data= labels,
           classification_type=classification_type,
           num_classes=n_classes,
           seed=seed_i,
           test_size=test_size,
           n_br=n_br,
           num_threads=n_threads,
           n_trials=n_trials_optuna,
        )
        (model, metrics_i, y_test_le, y_pred, data_used, weighted_params) = classification
        metrics.append(metrics_i)
        # Save each metrics_i as a CSV file with the seed included in the name
        metrics_i.to_csv(os.path.join(save_dir, f'metrics_seed_{seed_i}.csv'))

    # done for seed in seeds
    # Create df with mean and std of all seeds
    rows = metrics[0].index
    cols = metrics[0].columns
    metrics_seeds = np.array([metrics[i].values for i in range(n_seeds)])
    metrics_mean = pd.DataFrame(np.mean(metrics_seeds, axis=0), index=rows, columns=cols)
    metrics_std = pd.DataFrame(np.std(metrics_seeds, axis=0), index=rows, columns=cols)
    # Save mean and std of metrics
    metrics_mean.to_csv(os.path.join(save_dir,'metrics_mean.csv'))
    metrics_std.to_csv(os.path.join(save_dir,'metrics_std.csv'))
    # Boxplot of classification metrics:
    boxplot_metrics(plot_results=metrics,legend_labels=metrics_mean.columns,save_dir=save_dir,cmap='Pastel2',n_seeds=n_seeds,title=args.title)

if __name__ == '__main__':
    from datetime import datetime
    today = datetime.now().strftime("%Y%m%d")
    import argparse
    parser = argparse.ArgumentParser(description='Run classification on all cancers')
    parser.add_argument('--path_data',
                        type=str,
                        default='data/interim/20241023_preprocessing/rnaseq_maha.csv',
                        help='Path to the data file')
    parser.add_argument('--path_metadata',
                        type=str,
                        default='data/cavalli_subgroups.csv',
                        help='Path to the metadata file')
    parser.add_argument('--save_dir',
                        type=str,
                        default=f'data/interim/{today}_all_cancers_classification')
    parser.add_argument('--n_classes',
                        type=int,
                        default=4,
                        help='Number of classes to classify')
    # parser.add_argument('--stages', type=str, default='early_late', choices=['early_late','i_ii_iii_iv','i_ii_iii'],
    #                     help='Type of stages to classify: early_late, i_ii_iii_iv, i_ii_iii')
    # parser.add_argument('--stage_classification', type=str, default='early_late', choices=['early_late','i_ii_iii_iv','i_ii_iii'],
    #                     help='Type of stage classification')
    parser.add_argument('--classification_type',
                        type=str,
                        default='weighted',
                        choices=['weighted', 'unbalanced'],
                        help='Type of classification to perform')
    parser.add_argument('--n_seeds',
                        type=int,
                        default=10,
                        help='Number of seeds to run classification')
    parser.add_argument('--n_trials_optuna',
                        type=int,
                        default=100,
                        help='Number of optuna trials to run')
    parser.add_argument('--n_br',
                        type=int,
                        default=100,
                        help='Number of boosting rounds for xgboost')
    # parser.add_argument('--per', type=str, default=20,
    #                     help='Percentage of zeros in rnaseq needed to remove genes during preprocessing')
    parser.add_argument('--test_size',
                        type=float,
                        default=0.2,
                        help='Test size for classification')
    parser.add_argument('--n_threads',
                        type=int,
                        default=16,
                        help='Number of threads to use for parallel processing')
    parser.add_argument('--title',
                        type=str,
                        default='',
                        help='Title of the classification boxplot')
    # unpack arguments
    args = parser.parse_args()
    main(path_data=args.path_data,
         path_metadata=args.path_metadata,
         n_classes=args.n_classes,
         n_seeds=args.n_seeds,
         n_trials_optuna=args.n_trials_optuna,
         n_br=args.n_br,
         classification_type=args.classification_type,
         test_size=args.test_size,
         n_threads=args.n_threads,
         save_dir=args.save_dir,
         )