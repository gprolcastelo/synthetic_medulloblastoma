import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.covariance import MinCovDet
from pickle import dump
import scipy.stats
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# # Load data
def load_data(data_path,metadata_path):
    data = pd.read_csv(data_path, index_col=0)
    # Cancer groups:
    metadata = pd.read_csv(metadata_path, index_col=0).squeeze()
    # Check first dimension coincides:
    if data.shape[0] != metadata.shape[0]:
        data = data.T
    # Check first dimension coincides:
    assert data.shape[0] == metadata.shape[0], "First dimension of data and metadata do not coincide"
    # Set data patients to be the same as the groups index:
    data.index = metadata.index

    return data, metadata

def get_g3g4(data,groups):
    clinical_g3g4=groups[groups.isin(['Group3','Group4'])]
    data_g3g4 = data.loc[clinical_g3g4.index]
    return data_g3g4

def plot_original_distribution(data,save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(data.values.flatten(),bins=100,color="orange",ec="k",label="cavalli")
    ax.set_xlabel("Gene Expression",fontsize=12)
    ax.set_ylabel("Counts",fontsize=12)
    # plt.title(
    #     """
    #     Distribution of Gene Expression in Cavalli
    #     """,
    #     fontsize=14)
    # plt.legend()
    plt.savefig(os.path.join(save_path, 'original_distribution.png'), dpi=600)
    plt.savefig(os.path.join(save_path, 'original_distribution.svg'), dpi=600)
    plt.savefig(os.path.join(save_path, 'original_distribution.pdf'), dpi=600)
    plt.clf()
    # Creating var histogram:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(np.var(data, axis=1).values, bins=100, color="green", ec="k")
    # ax.vlines(0.1,0,4000,"r",lw=2)
    ax.set_xlabel("Variance of Genes' Expression", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    # ax.set_title("Distribution of Variance of Genes' Expression\nCavalli",fontsize=14)
    plt.savefig(os.path.join(save_path, 'original_variance_distribution.png'), dpi=600)
    plt.savefig(os.path.join(save_path, 'original_variance_distribution.svg'), dpi=600)
    plt.savefig(os.path.join(save_path, 'original_variance_distribution.pdf'), dpi=600)
    plt.clf(); fig.clf()
    # Creating mean histogram
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(np.mean(data, axis=1).values, bins=100, color="cornflowerblue", ec="k")
    # ax.vlines(0.5,0,1800,"r",lw=2)
    ax.set_xlabel("Mean of Genes' Expression", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    # ax.set_title("Distribution of Mean of Genes' Expression\nCavalli",fontsize=14)
    plt.savefig(os.path.join(save_path, 'original_mean_distribution.png'), dpi=600)
    plt.savefig(os.path.join(save_path, 'original_mean_distribution.svg'), dpi=600)
    plt.savefig(os.path.join(save_path, 'original_mean_distribution.pdf'), dpi=600)
    plt.clf();fig.clf()

def preprocess(data,per=0.2,cutoff=0.1):
    # Keep genes with 0 expression in at least per% of the patients
    data = data[(data==0).sum(axis=1)/data.shape[1]<=per]
    # Filter genes by var
    data = data.iloc[(np.var(data, axis=1).values >= cutoff)]
    return data

def maha_outliers(data,save_path,alpha=0.05):
    # Mahalanobis
    nrows = data.shape[0]
    print("n rows = n genes =",nrows)
    # Define cutoff
    cutoff = scipy.stats.chi2.ppf(1-alpha,nrows - 1)
    with open(os.path.join(save_path, "alpha.txt"), 'w') as f:
        f.write(str(alpha))
    with open(os.path.join(save_path, "cutoff.txt"), 'w') as f:
        f.write(str(cutoff))
    # Minimum Covariance Determinant:
    print("running mincovdet")
    output_c = MinCovDet().fit(data)
    print("done mincovdet")
    dump(output_c,open(os.path.join(save_path, 'MCDoutput_cavalli.pkl'), 'wb'))
    # Mahalanobis distance:
    md_c = output_c.dist_
    # Identify outliers Cavalli:
    names_outliers_MH_c = np.where(md_c > cutoff)
    names_outliers_MH_c = np.transpose(names_outliers_MH_c)
    names_outliers_MH_c = [int(i) for i in names_outliers_MH_c]
    print("number of outliers:\t",len(names_outliers_MH_c))
    print("")
    # Mahalanobis distance:
    md_c = output_c.dist_
    del output_c
    # Drop outliers from datset:
    data_maha = data.drop(data.index[names_outliers_MH_c])
    print("shape of output dataset:", data_maha.shape)
    print("")
    # Plot resulting distribution
    plt.figure(figsize=(7,5))
    plt.hist(data.values.flatten(),bins=100,color="green",ec="k",label="before")
    plt.hist(data_maha.values.flatten(),bins=100,color="cornflowerblue",ec="k",label="after")
    plt.xlabel("Gene Expression",fontsize=14)
    plt.ylabel("Counts",fontsize=14)
    # plt.title(
    #     """
    #     Distribution of Gene Expression,
    #     for all genes and patients,
    #     before and after Mahalanobis outlier detection
    #     Cavalli
    #     """,
    #     fontsize=14)
    plt.legend()
    # Save the figure
    plt.savefig(os.path.join(save_path, 'distribution_after_mahalanobis.png'), dpi=600)
    plt.savefig(os.path.join(save_path, 'distribution_after_mahalanobis.svg'), dpi=600)
    plt.savefig(os.path.join(save_path, 'distribution_after_mahalanobis.pdf'), dpi=600)

    return data_maha


def main(args):
    print('Path to save data:',args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    data, groups = load_data(data_path=args.data_path,metadata_path=args.metadata_path)
    print('Shape of original data:',data.shape)
    plot_original_distribution(data,args.save_path)
    # Check for null or missing data:
    print('null in data:',data.isnull().sum().sum())
    print('na in data:',data.isna().sum().sum())
    # Check all values in the df are real:
    print('Are all values in the data real?','Yes' if data.size==np.sum(np.isreal(data)) else 'No')
    # Get data for groups 3 and 4, before any preprocessing
    data_g3g4 = get_g3g4(data=data,groups=groups)
    data_g3g4.to_csv(os.path.join(args.save_path, 'g3g4_noprepro.csv'))
    # Preprocess data
    # Here the data must reflect samples as columns:
    data_preprocessed = preprocess(data=data.T,per=args.per,cutoff=args.cutoff)
    # Mahalanobis
    data_maha=maha_outliers(data=data_preprocessed,save_path=args.save_path,alpha=args.alpha)
    data_maha.T.to_csv(os.path.join(args.save_path, 'cavalli_maha.csv'))
    # Get data for groups 3 and 4, after preprocessing
    data_g3g4_maha = get_g3g4(data=data_maha.T,groups=groups)
    data_g3g4_maha.to_csv(os.path.join(args.save_path, 'g3g4_maha.csv'))
    print('done preprocessing')



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--metadata_path', type=str, help='path to metadata')
    parser.add_argument('--save_path', type=str, help='path to save results')
    parser.add_argument('--per', type=float, help='Filter out genes when they have this percentage of samples with 0 expression',default=0.2)
    parser.add_argument('--cutoff', type=float, help='Filter out genes with variance below this value',default=0.1)
    parser.add_argument('--alpha', type=float, help='alpha for Mahalanobis',default=0.05)
    args = parser.parse_args()
    plt.style.use('ggplot')
    main(args)