import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from src.models.train_model import set_seed
from src.models.my_model import VAE, CVAE
from src.utils import apply_VAE, get_hyperparams
from sklearn.model_selection import train_test_split
import optuna
plt.style.use('ggplot')

def load_data(full_data_path,clinical_data_path):
    data = pd.read_csv(full_data_path, index_col=0)
    clinical = pd.read_csv(clinical_data_path, index_col=0)
    clinical = clinical['Sample_characteristics_ch1']
    # clinical.replace({'Group3': 'Group 3', 'Group4': 'Group 4'}, inplace=True)
    return data, clinical

def load_model(model_path,model_here,data,classes,device,seed=2023,cvae=False):
    # check shape[0] of data is the same as classes
    if data.shape[0] != len(classes):
        print("Input data transposed because the number of samples in the data and classes are not the same")
        data = data.T
    # assert data is torch.tensor, if not convert it
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data.values).to(torch.float32)
    # Get model hyperparameters:
    idim, md, feat = get_hyperparams(model_path)
    if cvae:
        print('CVAE model')
        # unique number of classes
        num_classes = len(classes.unique())
        # One-hot encode the classes
        y = pd.get_dummies(classes).values
        print('idim =', idim, 'md =', md, 'feat =', feat,'num_classes =', num_classes)
    else:
        print('VAE model')
        print('idim =', idim, 'md =', md, 'feat =', feat)
        y=None
    # One hot encode the classes data:
    clinical_onehot = pd.get_dummies(classes)
    # Importing the model:
    set_seed(seed)
    if cvae:
        model_vae = model_here(idim, md, feat, num_classes)  # Initialize the model
    else:
        model_vae = model_here(idim, md, feat)  # Initialize the model
    model_vae.load_state_dict(torch.load(model_path, map_location=device))  # Load the state dictionary
    model_vae.eval()  # Set the model to evaluation mode
    # Apply AE or VAE to all data:
    reconstruction, _, _, z, scaler = apply_VAE(data=data,model_here=model_vae,y=y)
    return reconstruction, z, clinical_onehot, model_vae, scaler

def as_dataloader(data_train,data_test,batch_size):
    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
    )
    loader_test = torch.utils.data.DataLoader(
        data_test,
        batch_size=batch_size,
        shuffle=False,
    )
    return loader_train, loader_test

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, data_rec, data):
        self.data_rec = data_rec
        self.data = data

    def __len__(self):
        return len(self.data_rec)

    def __getitem__(self, idx):
        if idx >= len(self.data_rec) or idx >= len(self.data):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset with lengths {len(self.data_rec)} and {len(self.data)}"
            )

        return self.data_rec[idx], self.data[idx]

def as_paired_dataloader(data_rec, data, batch_size, shuffle=True, seed=42):
    dataset = PairedDataset(data_rec, data)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator
    )
    return loader

class NetworkReconstruction(nn.Module):
    def __init__(self, layer_dims):
        super(NetworkReconstruction, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_network_reconstruction(network, reconstruction,data,loader_train, loader_test, epochs, lr, device):
    loss_train = []
    loss_test = []

    network.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    for epoch in range(epochs):
        # Set network to training mode
        network.train()
        running_loss = 0.0
        for i in loader_train:
            data_rec = reconstruction.loc[i]
            data_here = data.loc[i]
            # to torch tensor float32
            data_rec = torch.tensor(data_rec.values).float()
            data_here = torch.tensor(data_here.values).float()

            data_rec = data_rec.to(device)
            data_here = data_here.to(device)
            optimizer.zero_grad()
            output = network(data_rec)
            loss = criterion(output, data_here)
            # Add running loss
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        # Append the average loss for the epoch
        loss_train.append(running_loss / len(loader_train))

        # Evaluate the network on the test set
        # Set network to evaluation mode
        network.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i in loader_test:
                data_rec = reconstruction.loc[i]
                data_here = data.loc[i]
                # to torch tensor float32
                data_rec = torch.tensor(data_rec.values).float()
                data_here = torch.tensor(data_here.values).float()
                data_rec = data_rec.to(device)
                data_here = data_here.to(device)
                output = network(data_rec)
                loss = criterion(output, data_here)
                running_loss += loss.item()
            # Append the average loss for the epoch
            loss_test.append(running_loss / len(loader_test))

        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}")
    return network, loss_train, loss_test

def plot_losses(loss_train, loss_test,save_path=None):
    plt.plot(loss_train, label='Train')
    plt.plot(loss_test, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)

def objective(trial,idim,df_reconstruction,data,loader_index_train, loader_index_test,args,device):
    # Define the hyperparameters to optimize
    input_dim = idim
    layer1_dim = trial.suggest_int('layer1_dim', 1024, 4096)
    layer2_dim = trial.suggest_int('layer2_dim', 128, 1024)
    layer3_dim = trial.suggest_int('layer3_dim', 1024, 4096)
    output_dim = idim

    network_dims = [input_dim, layer1_dim, layer2_dim, layer3_dim, output_dim]
    network = NetworkReconstruction(network_dims)

    # Train the network
    network, loss_train, loss_test = train_network_reconstruction(
        network, df_reconstruction, data, loader_index_train, loader_index_test,
        epochs=args.epochs, lr=args.lr, device=device
    )

    # Return the validation loss for the last 20 epochs as the objective value
    objective_value = np.mean(loss_test[-20:])
    print(f"Objective value: {objective_value} at trial {trial.number} with params {trial.params}")
    return objective_value

def main(args):
    """
    This script uses a deep neural network to adjust the reconstruction of the VAE model.
    We
    :return:
    """

    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data
    data, clinical = load_data(args.data_path, args.clinical_path)
    categs = sorted(clinical.unique())
    stage_to_int = {key:i for i,key in enumerate(categs)}
    clinical.replace(stage_to_int,inplace=True)
    # categs_int = sorted(clinical.unique())
    # Load model and get reconstruction
    if args.cvae:
        model_here = CVAE
    else:
        model_here = VAE
    reconstruction, z, clinical_onehot, model_vae, scaler=load_model(
        model_path=args.model_path,
        model_here=model_here,
        data=data,
        classes=clinical,
        device=device,
        seed=args.seed,
        cvae=args.cvae)
    df_reconstruction = pd.DataFrame(reconstruction, index=data.index, columns=data.columns)
    # One hot encode the classes data:
    # y = np.array(clinical).reshape(-1, 1) # Reshape the array
    # ohe = OneHotEncoder(categories=[categs_int], handle_unknown='ignore', sparse_output=False, dtype=np.int8).fit(y)
    # y = ohe.transform(y)
    # Split into train and test
    x_rec_train, x_rec_test, y_rec_train, y_rec_test = train_test_split(reconstruction, clinical, test_size=args.test_size, stratify=clinical,random_state=args.seed)
    x_data_train, x_data_test, _, _ = train_test_split(data, clinical, test_size=args.test_size, stratify=clinical,random_state=args.seed)
    # Get DataLoaders
    # partition = {'train': y_rec_train.index.tolist(), 'test': y_rec_test.index.tolist()}
    # labels = {key:i for i,key in zip(clinical.index.tolist(),clinical.values.tolist())}
    loader_index_train, loader_index_test = as_dataloader(y_rec_train.index, y_rec_test.index, batch_size=args.batch_size)
    # Hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial:
                   objective(trial,
                             x_rec_train.shape[1],
                             df_reconstruction,
                             data,
                             loader_index_train,
                             loader_index_test,
                             args,
                             device),
                   n_trials=args.n_trials
                   )

    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    df_best_hyperparams = pd.DataFrame([best_params])
    df_best_hyperparams.to_csv(os.path.join(args.output_path, "best_hyperparameters.csv"))
    # Train the network with the best hyperparameters
    network_dims = [
    x_rec_train.shape[1],
    best_params['layer1_dim'],
    best_params['layer2_dim'],
    best_params['layer3_dim'],
    x_rec_train.shape[1]
    ]
    network = NetworkReconstruction(network_dims)
    best_network, loss_train, loss_test = train_network_reconstruction(
        network, df_reconstruction, data, loader_index_train, loader_index_test,
        epochs=args.epochs, lr=args.lr, device=device
    )
    # Save the network
    os.makedirs(os.path.dirname(args.output_recnet_path), exist_ok=True) # create the output directory
    torch.save(network.state_dict(), args.output_recnet_path)
    print("Network saved to", args.output_recnet_path)
    # Save the losses
    pd.DataFrame(loss_train).to_csv(os.path.join(args.output_path, "loss_train.csv"))
    pd.DataFrame(loss_test).to_csv(os.path.join(args.output_path, "loss_test.csv"))
    # Plot the losses
    plot_losses(loss_train, loss_test, save_path=os.path.join(args.output_path, "losses.png"))
    return None


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser(description='Adjust the reconstruction of the VAE model')
    args.add_argument('--model_path', type=str, default=None)
    args.add_argument('--data_path', type=str, default=None)
    args.add_argument('--clinical_path', type=str, default=None)
    args.add_argument('--output_path', type=str, default=None)
    args.add_argument('--output_recnet_path', type=str, default=None)
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--test_size', type=float, default=0.2)
    args.add_argument('--seed', type=int, default=2023)
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--n_trials', type=int, default=100)
    args.add_argument('--cvae', action='store_true', help='Use CVAE model')
    args = args.parse_args()
    main(args=args)
    print('Done')