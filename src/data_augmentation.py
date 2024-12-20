"""
Data Augmentation
author: @gprolcastelo
date: 20240903
"""


# Necessary imports
import datetime
import os, sys, datetime, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
# Custom functions:
from src.models.train_model import set_seed
from src.models.my_model import VAE #, CVAE
from src.data_processing.shap import *
from src.data_processing.classification import *
from src.utils import apply_VAE, fun_decode, get_hyperparams, apply_recnet
from src.visualization.visualize import plot_umap
from src.adjust_reconstruction import NetworkReconstruction as model_net_here
plt.style.use('ggplot')

def str_or_list(value):
    """
    Function to check if the value is a string or a list. Returns as list with the string split by commas.
    :param value:
    :return:
    """
    if ',' in value:
        return [item.strip() for item in value.split(',')]
    else:
        return [value]

# Config:
def config(results_path):
    # Device for PyTorch
    device=torch.device('cpu')
    # Today's date
    today=datetime.datetime.now().strftime("%Y%m%d")
    # Dictionary with group colors for UMAP
    dict_umap = {
        # 'SHH': '#b22222',
        # 'WNT': '#6495ed',
        'Group 3': '#ffd700',
        'Group 4': '#008000',
        'G3-G4': '#db7093',
        'Synthetic': '#808080'
    }
    # Path to save the results
    # results_path = f"data/interim/{today}_data_augmentation"
    os.makedirs(results_path, exist_ok=True)
    print(results_path)
    return device, today, dict_umap, results_path

# Load data:
def load_data(full_data_path,clinical_data_path,model_path,network_model_path,recnet_hyperparams_path,seed=2023):
    rnaseq = pd.read_csv(full_data_path, index_col=0)
    print(rnaseq.shape)
    clinical = pd.read_csv(clinical_data_path, index_col=0)
    clinical = clinical['Sample_characteristics_ch1']
    # Make sure samples are the first dimension
    if rnaseq.shape[0] != clinical.shape[0]:
        rnaseq = rnaseq.T
    assert rnaseq.shape[0] == clinical.shape[0], ValueError('Data and metadata have different number of samples')
    print(clinical.shape)
    # Get idim, md, feat from the model path:
    idim, md, feat = get_hyperparams(model_path)
    print(idim, md, feat)
    # Importing the VAE model:
    set_seed(seed)
    model = VAE(idim, md, feat)  # Initialize the model
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the state dictionary
    model.eval()  # Set the model to evaluation mode
    # Import reconstruction network:
    # Hyperparameters:
    hyper = pd.read_csv(recnet_hyperparams_path, index_col=0)
    hyper = [idim] + hyper.values.tolist()[0] + [idim]
    # Importing the model:
    model_net = model_net_here(hyper)  # Initialize the model
    model_net.load_state_dict(
        torch.load(network_model_path, map_location=torch.device('cpu')))  # Load the state dictionary
    model_net.eval()  # Set the model to evaluation mode
    return rnaseq, clinical, idim, md, feat, model, model_net

def synth_data_generation(df_to_augment, metadata, group_to_augment, mu=0, std=1, n_synth=100, rand_state=2023, noise_ratio=0.1):
    df_augmented = df_to_augment.copy()
    metadata_augmented = metadata.copy()
    # Get patients in agg cluster
    for clu_i in group_to_augment:
        print('Augmenting group:', clu_i)
        print('metadata.value_counts() =', metadata.value_counts())
        real_patients = metadata[metadata.isin([clu_i])].index.to_list()
        print('len(real_patients) =', len(real_patients))
        # Df of real data with the patients in cluster i
        df_real_i = df_to_augment[df_to_augment.index.isin(real_patients)]
        print('df_real_i.shape =', df_real_i.shape)
        print('df_real_i min/max =', df_real_i.min().min(), df_real_i.max().max())
        # Initialize df of synthetic patients as a copy of group's real data
        df_synthetic_i = df_real_i.copy()
        # Add _synthetic to the index
        df_synthetic_i.index = [f'{patient}_synthetic_{clu_i}' for patient in df_synthetic_i.index]
        # Sample n_synth patients from df_z_synthetic with repetition only if all data is used
        replace = n_synth > len(df_synthetic_i)
        df_synthetic_sample = df_synthetic_i.sample(n=n_synth, replace=replace, random_state=rand_state)
        df_synthetic_sample.index = [f'synth_{i}_{clu_i}' for i in range(len(df_synthetic_sample.index))]
        # Add random noise to df_z_synthetic_sample
        np.random.seed(rand_state)
        noise = np.random.normal(mu, std, df_synthetic_sample.shape)
        # Calculate the scaling factor and shift
        min_val, max_val = np.min(df_synthetic_sample), np.max(df_synthetic_sample)
        # scale = (max_val - min_val) / (np.max(noise) - np.min(noise))
        # shift = min_val - np.min(noise) * scale
        # print('scale =', scale)
        # print('shift =', shift)
        if min_val < 0:
            noise_range = (noise_ratio * min_val, noise_ratio * max_val)
        else:
            noise_range = (-noise_ratio * max_val, noise_ratio * max_val)
        print('noise_range =', noise_range)
        # Scale noise to 0-1 range to match the data
        noise_scaled = MinMaxScaler(feature_range=noise_range).fit_transform(noise)
        print('noise_scaled min/max =', noise_scaled.min(), noise_scaled.max())
        # print('noise_scaled * scale + shift min/max =', (noise_scaled * scale + shift).min(), (noise_scaled * scale + shift).max())
        df_synthetic_sample += noise_scaled
        print('df_synthetic_sample.shape =', df_synthetic_sample.shape)
        print('df_synthetic_sample min/max =', df_synthetic_sample.min().min(), df_synthetic_sample.max().max())
        # Concat df_augmented with df_synthetic_sample
        df_augmented = pd.concat([df_augmented, df_synthetic_sample])
        # Add 'synthetic' to metadata for the synthetic patients
        metadata_augmented = pd.concat(
            [metadata_augmented,
             pd.Series({f'{patient}': f'synthetic_{clu_i}'
                        for patient in df_synthetic_sample.index.to_list()})
             ]
        )
        print('*' * 50 + '\n')

    print('df_augmented.shape =', df_augmented.shape)
    print('metadata_augmented.shape', metadata_augmented.shape)

    return df_augmented, metadata_augmented


def main(args):
    # Set up:
    device, today, dict_umap, results_path = config(args.results_path)

    # Load data:
    rnaseq, clinical, idim, md, feat, model_vae, model_net = load_data(args.data_path, args.clinical_path, args.model_path,args.network_model_path,args.recnet_hyperparams_path)

    # Apply VAE to rnaseq data:
    reconstruction_rnaseq, _, _, z, scaler = apply_VAE(torch.tensor(rnaseq.values).to(torch.float32), model_vae)
    # Convert z to a dataframe
    df_z = pd.DataFrame(z, index=rnaseq.index)
    # Augment selected group:
    df_z_rnaseq_augmented, clinical_synthetic = synth_data_generation(
        df_to_augment=df_z,
        metadata=clinical,
        group_to_augment=args.group_to_augment,
        mu=args.mu, std=args.std,
        n_synth=args.n_synth, rand_state=2023,
        noise_ratio=args.noise_ratio
    )
    # UMAP of the augmented data in the latent space
    clinical_for_umap = clinical_synthetic.copy()
    for group_i in args.group_to_augment:
        clinical_for_umap.replace({f'synthetic_{group_i}': 'Synthetic'},inplace=True)
    plot_umap(df_z_rnaseq_augmented,
              clinical_for_umap,
              dict_umap,
              n_components=2,
              save_fig=True,
              save_as=os.path.join(results_path, 'latent_augmented'),
              seed=None, title=None,show=False)
    # Apply decoder to the synthetic patients
    tensor_augmented = fun_decode(df_z_rnaseq_augmented, model_vae, scaler)
    print('tensor_augmented.shape =', tensor_augmented.shape)
    # Apply reconstruction network to the synthetic patients
    tensor_recnet = apply_recnet(data=tensor_augmented,model_net=model_net,seed=2023)
    df_net_output = pd.DataFrame(tensor_recnet.detach().numpy(),
                                 index=df_z_rnaseq_augmented.index,
                                 columns=rnaseq.columns)
    print('df_net_output.shape =', df_net_output.shape)
    # UMAP of the augmented data after reconstruction network
    plot_umap(df_net_output,
              clinical_for_umap,
              dict_umap,
              n_components=2,
              # save_fig=False,
              # save_as=None,
              save_fig=True,
              save_as=os.path.join(results_path, 'recnet_augmented'),
              seed=None, title=args.title,show=False)
    # Save the results
    # Create df with original patients and only the augmented patients after postprocessing
    df_augment_real_and_synth = rnaseq.copy()
    synth_pats = clinical_synthetic.copy()
    # synth_pats = clinical_synthetic[clinical_synthetic == f'synthetic_{args.group_to_augment}'].index
    synth_pats = clinical_synthetic[clinical_synthetic.isin([f'synthetic_{group}' for group in args.group_to_augment])].index
    df_augment_real_and_synth = pd.concat([df_augment_real_and_synth, df_net_output.loc[synth_pats]], axis=0)
    print('df_augment_real_and_synth.shape =', df_augment_real_and_synth.shape)
    df_augment_real_and_synth.to_csv(os.path.join(results_path, 'augmented_data.csv'))
    clinical_synthetic.to_csv(os.path.join(results_path, 'augmented_clinical.csv'))
    print('clinical_synthetic.shape =', clinical_synthetic.shape)
    df_z_rnaseq_augmented.to_csv(os.path.join(results_path, 'latent_augmented.csv'))
    print('df_z_rnaseq_augmented.shape =', df_z_rnaseq_augmented.shape)
    print('Results saved to:', results_path)

if __name__ == '__main__':

    # Get parameters from the command line
    import argparse

    parser = argparse.ArgumentParser(description='Data Augmentation')
    parser.add_argument('--data_path', type=str,
                        default='data/interim/20240301_Mahalanobis/cavalli.csv',
                        help='Path to the data')
    parser.add_argument('--clinical_path', type=str,
                        default='data/interim/20240801_clustering_g3g4/metadata_after_bootstrap.csv',
                        help='Path to the clinical data')
    parser.add_argument('--model_path', type=str,
                        default='models/20240417_cavalli_maha/20240417_VAE_idim12490_md2048_feat16mse_relu.pth',
                        help='Path to the model')
    parser.add_argument('--network_model_path', type=str,
                        default='data/interim/20240802_adjust_reconstruction/network_reconstruction.pth',
                        help='Path to the network model')
    parser.add_argument('--recnet_hyperparams_path', type=str, help='Path to the reconstruction network hyperparameters')
    parser.add_argument('--mu', type=float, default=0, help='Mean of the noise')
    parser.add_argument('--std', type=float, default=1, help='Standard deviation of the noise')
    parser.add_argument('--noise_ratio', type=float, default=0.25, help='Noise ratio')
    parser.add_argument('--group_to_augment', type=str_or_list, default='G3-G4', help='Group to augment')
    parser.add_argument('--n_synth', type=int, default=100, help='Number of synthetic patients to generate')
    parser.add_argument('--results_path', type=str, default='data/interim/20240903_data_augmentation', help='Path to save the results')
    parser.add_argument('--title', type=str, default='', help='Title of the UMAP')
    # parse the arguments
    args = parser.parse_args()
    print('args =', args)
    # Run the main function
    print('group_to_augment =', args.group_to_augment)
    main(args)
# Example usage:
# python data_augmentation.py --data_path data/interim/20240301_Mahalanobis/cavalli.csv \
#                             --clinical_path data/interim/20240801_clustering_g3g4/metadata_after_bootstrap.csv \
#                             --model_path models/20240417_cavalli_maha/20240417_VAE_idim12490_md2048_feat16mse_relu.pth \
#                             --network_model_path data/interim/20240802_adjust_reconstruction/network_reconstruction.pth \
#                             --mu 0 --std 1 --noise_ratio 0.25 --group_to_augment 'G3-G4' --n_synth 100
