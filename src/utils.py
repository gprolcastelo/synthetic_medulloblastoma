import re, torch
from sklearn.preprocessing import MinMaxScaler
from src.models.train_model import set_seed
import pandas as pd
import numpy as np

def apply_VAE(data,model_here,y=None):
    #model_here = load_model()
#     print("entered apply_VAE")
    #model_here = load_model()
    #global StandardScaler, MinMaxScaler
    #############################################################################
    # NORMALIZATION:
    # MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(data)
    data2 = scaler.transform(data)
    #############################################################################
#     print("torch no grad")
    with torch.no_grad():
        if y is None:
            data_latent, mu, logvar, z = model_here(torch.tensor(data2).float())
            # DE-NORMALIZE DATA:
            data_vae = scaler.inverse_transform(data_latent)
            #return data_vae,mu, logvar, z, None, scaler
        else:
            data_latent, mu, logvar, z = model_here(torch.tensor(data2).float(),torch.tensor(y).float())
            # DE-NORMALIZE DATA:
            data_vae = scaler.inverse_transform(data_latent)
            #return data_vae,mu, logvar, z, condition, scaler
    return data_vae, mu, logvar, z, scaler

def check_data(data):
    # check if data is DataFrame or numpy array
    if isinstance(data, pd.DataFrame):
        data = torch.tensor(data.values).float()
    elif isinstance(data, np.ndarray):
        data = torch.tensor(data).float()
    elif isinstance(data, torch.Tensor):
        data = data.float()
    else:
        raise ValueError("data is neither a pandas DataFrame, a numpy array, nor a torch tensor")
    return data

def apply_recnet(data,model_net,seed=2023):
    # check if data is DataFrame or numpy array
    data = check_data(data)
    # Importing the model:
    set_seed(seed)
    return model_net(data)

# We will use the encoder and reparametrization trick from the VAE
class EncoderWithReparam(torch.nn.Module):
    def __init__(self, vae_model, selected_lv=None):
        super().__init__()
        self.vae_model = vae_model
        self.selected_lv = selected_lv

    def forward(self, x):
        mu_logvar = self.vae_model.encoder(x).view(-1, 2, self.vae_model.features)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.vae_model.reparametrize(mu, logvar)
        if self.selected_lv is not None:
            z = z[:, self.selected_lv]
        return z


def fun_decode(data_z, model, scaler):
    # check if data_z is DataFrame or numpy array
    data_z = check_data(data_z)
    with torch.no_grad():
        data_recons = model.decoder(data_z)
    # DE-NORMALIZE DATA:
    return scaler.inverse_transform(data_recons)

def get_hyperparams(model_path):
    # Get Hyperparameters from model_path:
    # If 'md' is in the model_path, then it is not 3layers
    if 'md' in model_path:
        # Define the pattern to extract idim, md, and feat
        pattern = r"idim(\d+)_md(\d+)_feat(\d+)"
        # Search for the pattern in the model_path
        match = re.search(pattern, model_path)
        if match:
            idim = int(match.group(1))
            md = int(match.group(2)) if 'md' in model_path else None
            feat = int(match.group(3))
            return idim, md, feat
        else:
            print("Pattern not found in model_path")
    else:
        # Define the pattern to extract idim, feat
        pattern = r"idim(\d+)_feat(\d+)"
        # Search for the pattern in the model_path
        match = re.search(pattern, model_path)

        if match:
            idim = int(match.group(1))
            md = None
            feat = int(match.group(2))
            return idim, md, feat
        else:
            raise ValueError("Pattern not found in model_path")