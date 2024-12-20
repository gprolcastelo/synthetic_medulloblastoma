# Arguments
import argparse
parser = argparse.ArgumentParser(description='Check reconstruction with Wasserstein distance.')
parser.add_argument('--data_path', type=str, help='Path to the data file')
parser.add_argument('--clinical_path', type=str, help='Path to the clinical data file')
parser.add_argument('--model_path', type=str, help='Path to the model file')
parser.add_argument('--network_model_path', type=str, help='Path to the network model file')
parser.add_argument('--hyperparam_path', type=str, help='Path to the hyperparameters file')
parser.add_argument('--save_path', type=str, help='Path to save the figures')
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os, sys, re, time
import datetime
from tqdm import tqdm
from scipy.stats import wasserstein_distance
# Set the style to 'ggplot' for nicer looking plots
plt.style.use('ggplot')
# Custom functions:
from src.utils import apply_VAE, get_hyperparams
from src.models.train_model import *
from src.adjust_reconstruction import NetworkReconstruction as model_net_here

# Import model
cvae = False
if cvae:
    from src.models.my_model import CVAE as model_here
    num_classes = 4
else:
    from src.models.my_model import VAE as model_here
# Config:
device=torch.device('cpu')
today=datetime.now().strftime("%Y%m%d")


# Data

# Importing the data:
full_data_path = args.data_path
clinical_data_path = args.clinical_path

# Load data:
rnaseq = pd.read_csv(full_data_path, index_col=0)
clinical = pd.read_csv(clinical_data_path, index_col=0)
clinical = clinical['Sample_characteristics_ch1']

X_train, X_test, y_train, y_test=split_data(full_data_path,clinical_data_path,seed=2023)

# Apply one-hot encoding to the stages:
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y_train_onehot = enc.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
y_test_onehot = enc.transform(np.array(y_test).reshape(-1, 1)).toarray()

# # Model
model_path = args.model_path


# Hyperparameters:
idim, md, feat = get_hyperparams(model_path)

# Importing the model:
set_seed(2023)
if cvae:
    model = model_here(idim,md,feat,num_classes)
else:
    model = model_here(idim, md, feat)  # Initialize the model
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the state dictionary
model.eval()  # Set the model to evaluation mode

# Apply model to train data:
if cvae:
    reconstruction_x, _, _, z, _ = apply_VAE(torch.tensor(X_train.values).to(torch.float32), model, y_train_onehot)
else:
    reconstruction_x, _, _, z, _ = apply_VAE(torch.tensor(X_train.values).to(torch.float32), model)

df_reconstruction_x = pd.DataFrame(reconstruction_x, index=X_train.index, columns=X_train.columns)


# Apply model to test data:
set_seed(2023)
if cvae:
    reconstruction_x_test, _, _, z, scaler = apply_VAE(torch.tensor(X_test.values).to(torch.float32), model,y_test_onehot)
else:
    reconstruction_x_test, _, _, z, scaler = apply_VAE(torch.tensor(X_test.values).to(torch.float32), model)

df_reconstruction_x_test = pd.DataFrame(reconstruction_x_test, index=X_test.index, columns=X_test.columns)


# Reconstruction Network
save_path_rn = args.save_path
os.makedirs(save_path_rn, exist_ok=True)
network_model_path = args.network_model_path
hyperparam_path = args.hyperparam_path

# Hyperparameters:
hyper = pd.read_csv(hyperparam_path,index_col=0)
hyper=[idim]+hyper.values.tolist()[0]+[idim]



# Importing the model:
set_seed(2023)
model_net = model_net_here(hyper)  # Initialize the model
model_net.load_state_dict(torch.load(network_model_path, map_location=torch.device(device)))  # Load the state dictionary
model_net.eval()  # Set the model to evaluation mode

rec_tensor_test = torch.tensor(reconstruction_x_test).to(torch.float32)
net_output_test = model_net(rec_tensor_test)

df_net_output_test = pd.DataFrame(net_output_test.detach().numpy(), index=X_test.index, columns=X_test.columns)

# Wasserstein distance between all the original and reconstructed data for genes:
w_dists_genes = [wasserstein_distance(X_test[gene_i], df_net_output_test[gene_i])
                for gene_i in X_test.columns]

# Wasserstein distance between all the original and reconstructed patients:
w_dists_patients = [wasserstein_distance(X_test.loc[pat_i], df_net_output_test.loc[pat_i])
                    for pat_i in X_test.index]

# Wasserstein distance between all the original and VAE-reconstructed patients:
w_dists_patients_vae = [wasserstein_distance(X_test.loc[pat_i], df_reconstruction_x_test.loc[pat_i])
                    for pat_i in X_test.index]

# Wasserstein distance between all the original and reconstructed data for genes:
# tuple of [model_i[gene_i]]
w_dists_genes_vae = [wasserstein_distance(X_test[gene_i], df_reconstruction_x_test[gene_i]) 
                for gene_i in X_test.columns]

# Plot the wasserstein distances for the genes:
plt.figure(figsize=(10, 6))
plt.errorbar(np.arange(len(w_dists_genes)), 
                w_dists_genes, 
                fmt='o', 
                ecolor='r', 
                capthick=2,
                mfc='cornflowerblue',
                mec='black',
                ms=5,
                mew=1)
plt.title('Wasserstein Distance per Gene.\nReconstruction Network Output')
plt.xlabel('Gene')
plt.ylabel('Wasserstein Distance')
plt.ylim([-0., 9])
plt.savefig(os.path.join(save_path_rn,'wasserstein_distance_genes_recnetwork_nooutliers.svg'),format='svg',bbox_inches='tight')
plt.savefig(os.path.join(save_path_rn,'wasserstein_distance_genes_recnetwork_nooutliers.png'),format='png',dpi=600,transparent=True,bbox_inches='tight')
plt.savefig(os.path.join(save_path_rn,'wasserstein_distance_genes_recnetwork_nooutliers.pdf'),format='pdf',bbox_inches='tight')

# Get gene names with wasserstein distances above 1:
genes_above_1 = [gene for gene, dist in zip(X_test.columns, w_dists_genes) if dist > 1]
len(genes_above_1), genes_above_1
pd.DataFrame(genes_above_1).to_csv(os.path.join(save_path_rn,'genes_above_1.csv'))

# Plot the wasserstein distances for the genes:
plt.figure(figsize=(10, 6))
plt.errorbar(np.arange(len(w_dists_genes_vae)), 
                w_dists_genes_vae, 
                fmt='o', 
                ecolor='r', 
                capthick=2,
                mfc='cornflowerblue',
                mec='black',
                ms=5,
                mew=1)
plt.title('Wasserstein Distance per Gene.\n'+['VAE','CVAE'][cvae] + ' ' + 'Decoded')
plt.xlabel('Gene')
plt.ylabel('Wasserstein Distance')
plt.ylim([-0., 9])
plt.savefig(os.path.join(save_path_rn,'wasserstein_distance_genes_vae.svg'),format='svg',bbox_inches='tight')
plt.savefig(os.path.join(save_path_rn,'wasserstein_distance_genes_vae.png'),format='png',dpi=600,transparent=True,bbox_inches='tight')
plt.savefig(os.path.join(save_path_rn,'wasserstein_distance_genes_vae.pdf'),format='pdf',bbox_inches='tight')

# Plot the wasserstein distances for the patients
plt.figure(figsize=(10, 6))
plt.errorbar(np.arange(len(w_dists_patients)), 
                w_dists_patients, 
                fmt='o', 
                ecolor='r', 
                capthick=2,
                mfc='cornflowerblue',
                mec='black',
                ms=5,
                mew=1)
plt.title('Wasserstein Distance per Patient.\nReconstruction Network Output')
plt.xlabel('Patient')
plt.ylabel('Wasserstein Distance')
plt.ylim([0., 0.3])
plt.savefig(os.path.join(save_path_rn,'wasserstein_distance_patients_recnetwork.svg'),format='svg',bbox_inches='tight')
plt.savefig(os.path.join(save_path_rn,'wasserstein_distance_patients_recnetwork.png'),format='png',dpi=600,transparent=True,bbox_inches='tight')
plt.savefig(os.path.join(save_path_rn,'wasserstein_distance_patients_recnetwork.pdf'),format='pdf',bbox_inches='tight')

# Plot the wasserstein distances for the patients from VAE
plt.figure(figsize=(10, 6))
plt.errorbar(np.arange(len(w_dists_patients_vae)),
                w_dists_patients_vae,
                fmt='o',
                ecolor='r',
                capthick=2,
                mfc='cornflowerblue',
                mec='black',
                ms=5,
                mew=1)
plt.title('Wasserstein Distance per Patient.\n'+['VAE','CVAE'][cvae]+' Decoded')
plt.xlabel('Patient')
plt.ylabel('Wasserstein Distance')
plt.ylim([0., 0.3])
plt.savefig(os.path.join(save_path_rn,'wasserstein_distance_patients_vae.svg'),format='svg',bbox_inches='tight')
plt.savefig(os.path.join(save_path_rn,'wasserstein_distance_patients_vae.png'),format='png',dpi=600,transparent=True,bbox_inches='tight')
plt.savefig(os.path.join(save_path_rn,'wasserstein_distance_patients_vae.pdf'),format='pdf',bbox_inches='tight')



# # Checking several models
# print('Checking several models')
# # wait 10 seconds
# time.sleep(10)
# # Folder containing models:
# models_folder = "models/20241115_VAE"
#
# # Get all the models in the folder:
# models_list = os.listdir(models_folder)
# # keep only models that finish with lr0.0001.pth
# models_list = [model for model in os.listdir(models_folder) if model.endswith('lr0.0001.pth')]
#
# from src.models.my_model import VAE as model_here
#
# # Get hyperparameters for all models:
# hyperparams = [get_hyperparams(os.path.join(models_folder, model_path)) for model_path in models_list]
# set_seed(2023)
# # set all models' hyperparameters:
# models = [model_here(idim, md,feat) for idim, md, feat in hyperparams]
# # load state dictionaries for all models:
# for model, model_path in zip(models, models_list):
#     model.load_state_dict(torch.load(os.path.join(models_folder, model_path),
#                                      map_location=torch.device('cpu'))
#                           )
# # Set all models to evaluation mode:
# models = [model.eval() for model in models]
#
# # Apply all models to test data:
# vae_on_test = [apply_VAE(torch.tensor(X_test.values).to(torch.float32),model) for model in tqdm(models)]
# reconstruction_x_test = [reconstruction_x_test for (reconstruction_x_test, _, _, _, _) in tqdm(vae_on_test)]
#
# # Turn all reconstructions into dataframes:
# df_reconstruction_x_test = [pd.DataFrame(reconstruction_x_test_i,
#                                          index=X_test.index, columns=X_test.columns)
#                             for reconstruction_x_test_i in tqdm(reconstruction_x_test)]
#
#
#
# # Apply all models to train data:
# vae_on_train = [apply_VAE(torch.tensor(X_train.values).to(torch.float32),model) for model in tqdm(models)]
# reconstruction_x_train = [reconstruction_x_train for (reconstruction_x_train, _, _, _, _) in tqdm(vae_on_train)]
#
# # Turn all reconstructions into dataframes:
# df_reconstruction_x_train = [pd.DataFrame(reconstruction_x_train_i,
#                                          index=X_train.index, columns=X_train.columns)
#                             for reconstruction_x_train_i in tqdm(reconstruction_x_train)]
#
#
# # # Wasserstein distances per model
#
# save_dir_fig = 'reports/figures/20241115_wasserstein_models_comparison/'
# os.makedirs(save_dir_fig, exist_ok=True)
#
# # ### Per patient
# # Wasserstein distance between all the original and reconstructed data for patient pat_i:
# # tuple of [model_i[pat_i]]
# w_dists_pats = [[wasserstein_distance(X_test.loc[pat_i], df_reconstruction_x_test_i.loc[pat_i])
#                 for pat_i in df_reconstruction_x_test[0].index]
#                 for df_reconstruction_x_test_i in tqdm(df_reconstruction_x_test)]
#
# # Wasserstein distance between all the original and reconstructed data for patient pat_i:
# # tuple of [model_i[pat_i]]
# w_dists_pats_train = [[wasserstein_distance(X_train.loc[pat_i], df_reconstruction_x_train_i.loc[pat_i])
#                 for pat_i in df_reconstruction_x_train[0].index]
#                 for df_reconstruction_x_train_i in tqdm(df_reconstruction_x_train)]
#
# # Wasserstein distance between all the original and reconstructed data for patient pat_i:
# # tuple of [model_i[pat_i]]
# w_dists_pats_test = [[wasserstein_distance(X_test.loc[pat_i], df_reconstruction_x_test_i.loc[pat_i])
#                 for pat_i in df_reconstruction_x_test[0].index]
#                 for df_reconstruction_x_test_i in tqdm(df_reconstruction_x_test)]
#
# # Calculate mean and std of wasserstein distances per patient in w_dists_pats:
# w_dists_pats_mean = np.nanmean(np.array(w_dists_pats), axis=0)
# w_dists_pats_std = np.nanstd(np.array(w_dists_pats), axis=0)
#
# w_dists_pats_mean_train = np.nanmean(np.array(w_dists_pats_train), axis=0)
# w_dists_pats_std_train = np.nanstd(np.array(w_dists_pats_train), axis=0)
#
#
# # Import Cavalli data
# cavalli_path = "data/raw/GEO/cavalli_subgroups.csv"
# clinical_cavalli = pd.read_csv(cavalli_path, index_col=0)
#
# # Plot mean and std of wasserstein distances per patient
# plt.figure(figsize=(10, 6))
# plt.errorbar(np.arange(len(w_dists_pats_mean_train)),
#              w_dists_pats_mean_train,
#              yerr=w_dists_pats_std_train,
#              fmt='none',
#              ecolor='r',
#              capthick=2)
# plt.scatter(np.arange(len(w_dists_pats_mean_train)),
#             w_dists_pats_mean_train,
#             linewidths=1,
#             zorder=3,
#             edgecolor='black')
# plt.xlabel('Patient')
# plt.ylabel('Wasserstein Distance')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_patients_nc_colors_train.svg'),format='svg')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_patients_nc_colors_train.png'),format='png',dpi=600,transparent=True,bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_patients_nc_colors_train.pdf'),format='pdf',bbox_inches='tight')
#
# # Plot mean and std of wasserstein distances per patient
# plt.figure(figsize=(10, 6))
# plt.errorbar(np.arange(len(w_dists_pats_mean)),
#              w_dists_pats_mean,
#              yerr=w_dists_pats_std,
#              fmt='none',
#              ecolor='r',
#              capthick=2)
# plt.scatter(np.arange(len(w_dists_pats_mean)),
#             w_dists_pats_mean,
#             facecolor= 'cornflowerblue',
#             linewidths=1,
#             zorder=3,
#             edgecolor='black')
# plt.xlabel('Patient')
# plt.ylabel('Wasserstein Distance')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_patients_nc_colors_test.svg'),format='svg')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_patients_nc_colors_test.png'),format='png',dpi=600,transparent=True,bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_patients_nc_colors_test.pdf'),format='pdf',bbox_inches='tight')
#
# # Plot mean and std of wasserstein distances per patient:
# plt.errorbar(np.arange(len(w_dists_pats_mean)),
#              w_dists_pats_mean,
#              yerr=w_dists_pats_std,
#              fmt='o',
#              c ='cornflowerblue',
#              mec= 'k',
#              ecolor='r',
#              capthick=2)
# plt.title('Wasserstein Distance per Model\nPatients')
# plt.xlabel('Patient')
# plt.ylabel('Wasserstein Distance')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_patients.svg'),format='svg')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_patients.png'),format='png',dpi=600,transparent=True,bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_patients.pdf'),format='pdf',bbox_inches='tight')
#
# # Wasserstein distance between all the original and reconstructed data for patient pat_i:
# # tuple of [model_i[pat_i]]
# w_dists_pats_train = [[wasserstein_distance(X_train.loc[pat_i], df_reconstruction_x_train_i.loc[pat_i])
#                 for pat_i in df_reconstruction_x_train[0].index]
#                 for df_reconstruction_x_train_i in tqdm(df_reconstruction_x_train)]
#
# # Wasserstein distance between all the original and reconstructed data for gene gene_i:
# # tuple of [model_i[gene_i]]
# w_dists_genes_train = [[wasserstein_distance(X_test[gene_i], df_reconstruction_x_test_i[gene_i])
#                 for gene_i in df_reconstruction_x_test[0].columns]
#                 for df_reconstruction_x_test_i in tqdm(df_reconstruction_x_test)]
#
# # Calculate mean and std of wasserstein distances per model in w_dists_pats_train:
# w_dists_pats_train_mean = np.array(w_dists_pats_train).mean(axis=1)
# w_dists_pats_train_std = np.array(w_dists_pats_train).std(axis=1)
#
# plt.figure(figsize=(10, 6))
# # Plot mean and std of wasserstein distances per model:
# plt.errorbar(np.arange(len(w_dists_pats_train_mean)),
#              w_dists_pats_train_mean,
#              yerr=w_dists_pats_train_std,
#              fmt='o',
#              ecolor='r',
#              capthick=2)
# # Highlight the models with the lowest mean wasserstein distance with a yellow dot:
# plt.scatter(np.array(w_dists_pats_train_mean).argsort()[0],
#             w_dists_pats_train_mean[np.array(w_dists_pats_train_mean).argmin()],
#             color='yellow',
#             label='Lowest Wasserstein',
#             zorder=3)
# # Highlight the models with the second lowest mean wasserstein distance with a green dot:
# plt.scatter(np.array(w_dists_pats_train_mean).argsort()[1],
#             w_dists_pats_train_mean[np.array(w_dists_pats_train_mean).argsort()[1]],
#             color='green',
#             label='Second Lowest Wasserstein',
#             zorder=3)
# # plt.title('Average Wasserstein Distance between Patients per Model.\nTrain Data.')
# plt.xlabel('Model')
# plt.ylabel('Wasserstein Distance')
# # Set xticks to hyperparameters:
# plt.xticks(np.arange(len(w_dists_pats_train_mean)), hyperparams, rotation=90,fontsize=12)
# plt.savefig(os.path.join(save_dir_fig,'avg_wasserstein_distance_patients_train.svg'),
#             format='svg',
#             bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'avg_wasserstein_distance_patients_train.png'),
#             format='png',
#             dpi=600,
#             transparent=True,
#             bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'avg_wasserstein_distance_patients_train.pdf'),
#             format = 'pdf',
#             bbox_inches='tight')
#
#
# # Calculate mean and std of wasserstein distances per model in w_dists_pats_test:
# w_dists_pats_test_mean = np.array(w_dists_pats_test).mean(axis=1)
# w_dists_pats_test_std = np.array(w_dists_pats_test).std(axis=1)
# # Plot mean and std of wasserstein distances per model:
# plt.figure(figsize=(10, 6))
# plt.errorbar(np.arange(len(w_dists_pats_test_mean)),
#              w_dists_pats_test_mean,
#              yerr=w_dists_pats_test_std,
#              fmt='o',
#              ecolor='r',
#              capthick=2)
# # Highlight the models with the lowest mean wasserstein distance with a yellow dot:
# plt.scatter(np.array(w_dists_pats_test_mean).argsort()[0],
#             w_dists_pats_test_mean[np.array(w_dists_pats_test_mean).argmin()],
#             color='yellow',
#             label='Lowest Wasserstein',
#             zorder=3)
# # Highlight the models with the second lowest mean wasserstein distance with a green dot:
# plt.scatter(np.array(w_dists_pats_test_mean).argsort()[1],
#             w_dists_pats_test_mean[np.array(w_dists_pats_test_mean).argsort()[1]],
#             color='green',
#             label='Second Lowest Wasserstein',
#             zorder=3)
# # plt.title('Average Wasserstein Distance between Patients per Model.\nTest Data.')
# plt.xlabel('Model')
# # Set xticks to hyperparameters:
# plt.xticks(np.arange(len(w_dists_pats_train_mean)), hyperparams, rotation=90,fontsize=12)
# plt.ylabel('Wasserstein Distance')
# # plt.legend(loc='upper right')
# plt.savefig(os.path.join(save_dir_fig,'avg_wasserstein_distance_patients_test.svg'),
#             format='svg',
#             bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'avg_wasserstein_distance_patients_test.png'),
#             format='png',
#             dpi=600,
#             transparent=True,
#             bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'avg_wasserstein_distance_patients_test.pdf'),
#             format = 'pdf',
#             bbox_inches='tight')
# # ### Per gene
#
# # Wasserstein distance between all the original and reconstructed data for genes:
# # tuple of [model_i[gene_i]]
# w_dists_genes = [[wasserstein_distance(X_test[gene_i], df_reconstruction_x_test_i[gene_i])
#                 for gene_i in df_reconstruction_x_test[0].columns]
#                 for df_reconstruction_x_test_i in tqdm(df_reconstruction_x_test)]
#
# # Calculate mean and std of wasserstein distances per gene in w_dists_genes:
# w_dists_genes_mean = np.nanmean(np.array(w_dists_genes),axis=0)
# w_dists_genes_std = np.nanstd(np.array(w_dists_genes),axis=0)
#
#
# # Plot mean and std of wasserstein distances per patient:
# plt.errorbar(np.arange(np.array(w_dists_genes).shape[1]),
#              w_dists_genes_mean,
#              yerr=w_dists_genes_std,
#              fmt='o',
#              ecolor='r',
#              capthick=2,
#              mfc='cornflowerblue',
#              mec='black',
#              ms=5,
#              mew=1)
# plt.title('Wasserstein Distance per Gene')
# plt.xlabel('Gene')
# plt.ylabel('Wasserstein Distance')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_genes.svg'),format='svg')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_genes.png'),format='png',dpi=600,transparent=True,bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_genes.pdf'),format='pdf',bbox_inches='tight')
#
#
#
# # Calculate mean and std of wasserstein distances per model in w_dists_genes_train:
# w_dists_genes_train_mean = np.nanmean(np.array(w_dists_genes_train), axis=1)
# w_dists_genes_train_std = np.nanstd(np.array(w_dists_genes_train), axis=1)
#
# plt.figure(figsize=(10, 6))
# # Plot mean and std of wasserstein distances per model:
# plt.errorbar(np.arange(len(w_dists_genes_train_mean)),
#              w_dists_genes_train_mean,
#              yerr=w_dists_genes_train_std,
#              mfc='cornflowerblue',
#              mec='black',
#              fmt='o',
#              ecolor='r',
#              capthick=2)
# # Highlight the models with the lowest mean wasserstein distance with a yellow dot:
# plt.scatter(np.array(w_dists_genes_train_mean).argsort()[0],
#             w_dists_genes_train_mean[np.array(w_dists_genes_train_mean).argmin()],
#             color='yellow',
#             edgecolors='black',
#             label='Lowest Wasserstein',
#             zorder=3)
# # Highlight the models with the second lowest mean wasserstein distance with a green dot:
# plt.scatter(np.array(w_dists_genes_train_mean).argsort()[1],
#             w_dists_genes_train_mean[np.array(w_dists_genes_train_mean).argsort()[1]],
#             color='green',
#             label='Second Lowest Wasserstein',
#             edgecolors='black',
#             zorder=3)
# plt.title('Average Wasserstein Distance between Genes per Model.\nTrain Data.')
# plt.xlabel('Model')
# plt.ylabel('Wasserstein Distance')
# # Set xticks to hyperparameters:
# plt.xticks(np.arange(len(w_dists_pats_train_mean)), hyperparams, rotation=90,fontsize=12)
# plt.legend(loc='upper right')
# plt.savefig(os.path.join(save_dir_fig,'avg_wasserstein_distance_genes_train.svg'),format='svg',bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'avg_wasserstein_distance_genes_train.png'),format='png',dpi=600,transparent=True,bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'avg_wasserstein_distance_genes_train.pdf'),format='pdf',bbox_inches='tight')
#
#
# # Calculate mean and std of wasserstein distances per model with genes:
# w_dists_genes_mean = np.nanmean(np.array(w_dists_genes), axis=1)
# w_dists_genes_std = np.nanstd(np.array(w_dists_genes), axis=1)
#
# plt.figure(figsize=(10, 6))
# # Plot mean and std of wasserstein distances per model:
# plt.errorbar(np.arange(len(w_dists_genes_mean)),
#              w_dists_genes_mean,
#              yerr=w_dists_genes_std,
#              fmt='o',
#              mfc='cornflowerblue',
#              mec='black',
#              ecolor='red',
#              capthick=2)
#
# # Highlight the models with the lowest mean wasserstein distance with a yellow dot:
# plt.scatter(np.array(w_dists_genes_mean).argmin(),
#             w_dists_genes_mean[np.array(w_dists_genes_mean).argmin()],
#             color='yellow',
#             label='Lowest Wasserstein',
#             edgecolors='black',
#             zorder=3)  # Add this line
#
# # Highlight the models with the second lowest mean wasserstein distance with a green dot:
# plt.scatter(np.array(w_dists_genes_mean).argsort()[1],
#             w_dists_genes_mean[np.array(w_dists_genes_mean).argsort()[1]],
#             color='green',
#             edgecolors='black',
#             label='Second Lowest Wasserstein',
#             zorder=3)  # Add this line
#
# plt.title('Average  Wasserstein Distance between Genes per Model.\nTest Data.')
# plt.xlabel('Model')
# plt.ylabel('Wasserstein Distance')
# # Set xticks to hyperparameters:
# plt.xticks(np.arange(len(w_dists_pats_train_mean)), hyperparams, rotation=90,fontsize=12)
# plt.savefig(os.path.join(save_dir_fig,'avg_wasserstein_distance_genes_test.svg'),format='svg',bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'avg_wasserstein_distance_genes_test.png'),format='png',dpi=600,transparent=True,bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'avg_wasserstein_distance_genes_test.pdf'),format='pdf',bbox_inches='tight')
#
# # Get parameters of models with lowest mean wasserstein distance:
# min_wasserstein_distance = np.nanargmin(np.array(w_dists_genes_mean))
#
# # Get parameters of model with second lowest mean wasserstein distance:
# second_min_wasserstein_distance = np.array(w_dists_genes_mean).argsort()[1]
#
# # ### Per gene
#
# # Wasserstein distance between all the original and reconstructed data for gene gene_i:
# # tuple of [model_i[gene_i]]
# w_dists_genes_train = [[wasserstein_distance(X_train[gene_i], df_reconstruction_x_train_i[gene_i])
#                 for gene_i in df_reconstruction_x_train[0].columns]
#                 for df_reconstruction_x_train_i in tqdm(df_reconstruction_x_train)]
#
# # Calculate mean and std of wasserstein distances per gene in w_dists_genes_train:
# w_dists_genes_train_mean = np.nanmean(np.array(w_dists_genes_train),axis=0)
# w_dists_genes_train_std = np.nanstd(np.array(w_dists_genes_train),axis=0)
#
# # Wasserstein distance between all the original and reconstructed data for genes:
# # tuple of [model_i[gene_i]]
# w_dists_genes_test = [[wasserstein_distance(X_test[gene_i], df_reconstruction_x_test_i[gene_i])
#                 for gene_i in df_reconstruction_x_test[0].columns]
#                 for df_reconstruction_x_test_i in tqdm(df_reconstruction_x_test)]
#
# # Calculate mean and std of wasserstein distances per gene in w_dists_genes:
# w_dists_genes_mean = np.nanmean(np.array(w_dists_genes_test),axis=0)
# w_dists_genes_std = np.nanstd(np.array(w_dists_genes_test),axis=0)
#
#
#
# # Plot mean and std of wasserstein distances per patient: Test Data
# plt.errorbar(np.arange(np.array(w_dists_genes_test).shape[1]),
#              w_dists_genes_mean,
#              yerr=w_dists_genes_std,
#              fmt='o',
#              ecolor='r',
#              capthick=2,
#              mfc='cornflowerblue',
#              mec='black',
#              ms=5,
#              mew=1)
# plt.title('Wasserstein Distance per Gene. Test Data.')
# plt.xlabel('Gene')
# plt.ylabel('Wasserstein Distance')
# plt.ylim((0,5))
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_genes_test.svg'),format='svg')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_genes_test.png'),format='png',dpi=600,transparent=True,bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_genes_test.pdf'),
#             format='pdf',
#             bbox_inches='tight')
#
#
# # Plot mean and std of wasserstein distances per patient: Train Data
# plt.errorbar(np.arange(np.array(w_dists_genes_train).shape[1]),
#              w_dists_genes_train_mean,
#              yerr=w_dists_genes_train_std,
#              fmt='o',
#              ecolor='r',
#              capthick=2,
#              mfc='cornflowerblue',
#              mec='black',
#              ms=5,
#              mew=1)
# plt.title('Wasserstein Distance per Gene. Train Data.')
# plt.xlabel('Gene')
# # plt.xlim((0,9000))
# plt.ylabel('Wasserstein Distance')
# plt.ylim((0,5))
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_genes_train.svg'),format='svg')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_genes_train.png'),format='png',dpi=600,transparent=True,bbox_inches='tight')
# plt.savefig(os.path.join(save_dir_fig,'wasserstein_distance_genes_train.pdf'),
#             format='pdf',
#             bbox_inches='tight')


print('Done!')