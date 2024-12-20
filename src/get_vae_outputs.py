import pandas as pd
import torch, os
from src.utils import apply_VAE, apply_recnet, get_hyperparams, check_data
from src.models.train_model import set_seed
from src.models.my_model import VAE
from src.adjust_reconstruction import NetworkReconstruction

def load_data(data_path,metadata_path):
    # Load the data
    data = pd.read_csv(data_path,index_col=0)
    metadata = pd.read_csv(metadata_path,index_col=0).squeeze()
    # Check first dimension coincides
    if data.shape[0] != metadata.shape[0]:
        data = data.T
    # assert
    assert data.shape[0] == metadata.shape[0], ValueError('Data and metadata have different number of samples')
    return data, metadata

def load_model(model_path,model,hyperparams,seed=2023):
    # Importing the model:
    set_seed(seed)
    if model.__name__ == 'VAE':
        idim, md, feat = hyperparams
        model_vae = model(input_dim=idim, mid_dim=md, features=feat)  # Initialize the model
    elif model.__name__ == 'NetworkReconstruction':
        model_vae = model(hyperparams)
    else:
        raise ValueError('Model not recognized. Must be VAE or NetworkReconstruction')
    model_vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the state dictionary
    model_vae.eval()  # Set the model to evaluation mode
    return model_vae

def main(args):
    # Load the data
    data, _ = load_data(data_path=args.data_path,metadata_path=args.metadata_path)
    # Load VAE
    idim, md, feat = get_hyperparams(model_path=args.model_path) # Get hyperparameters
    model_vae = load_model(model_path=args.model_path, model=VAE, hyperparams=(idim, md, feat), seed=args.seed) # Load the model with the hyperparameters
    # Apply AE or VAE to all data:
    data_for_model = check_data(data=data)
    decoded, _, _, z, scaler = apply_VAE(data=data_for_model, model_here=model_vae, y=None)
    df_decoded = pd.DataFrame(decoded, index=data.index, columns=data.columns)
    df_z = pd.DataFrame(z, index=data.index)
    # Load postprocessing reconstruction network
    hyper = pd.read_csv(args.recnet_hyperparams_path,index_col=0)
    hyper=[idim]+hyper.values.tolist()[0]+[idim]
    # hyper = [2664, 148, 3870]
    # hyper=[idim]+hyper+[idim]
    recnet = load_model(model_path=args.recnet_path, model=NetworkReconstruction, hyperparams=hyper, seed=args.seed)
    postprocessed = apply_recnet(data=df_decoded, model_net=recnet, seed=args.seed)
    postprocessed = postprocessed.detach().numpy() # Convert tensor to numpy
    df_postprocessed = pd.DataFrame(postprocessed, index=data.index, columns=data.columns)
    # Save the data
    os.makedirs(args.output_path,exist_ok=True)
    print('Saving data to:',args.output_path)
    print('Saving decoded data...')
    df_decoded.to_csv(os.path.join(args.output_path,'decoded.csv'))
    print('Saving latent space...')
    df_z.to_csv(os.path.join(args.output_path,'z.csv'))
    print('Saving postprocessed data...')
    df_postprocessed.to_csv(os.path.join(args.output_path,'postprocessed.csv'))
    print('Done!')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Apply VAE to data')
    parser.add_argument('--data_path', type=str, help='Path to the data')
    parser.add_argument('--metadata_path', type=str, help='Path to the metadata')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--recnet_path', type=str, help='Path to the reconstruction network')
    parser.add_argument('--recnet_hyperparams_path', type=str, help='Path to the reconstruction network hyperparameters')
    parser.add_argument('--output_path', type=str, help='Path to save the output')
    parser.add_argument('--seed', type=int, default=2023, help='Seed for reproducibility')
    args = parser.parse_args()
    main(args)


# Example usage:
# python src/get_vae_outputs.py --data_path data/interim/20240301_Mahalanobis/cavalli.csv \
#                               --metadata_path data/cavalli_subgroups.csv \
#                               --model_path models/20240417_cavalli_maha/20240417_VAE_idim12490_md2048_feat16mse_relu.pth \
#                               --recnet_path data/interim/20240802_adjust_reconstruction/network_reconstruction.pth \
#                               --recnet_hyperparams_path models/RecNet/RecNet_hyperparams.csv \
#                               --output_path data/processed/20241104_vae_output \
#                               --seed 2023
# TODO: adapt to new model, which has been trained with the new preprocessed data (>~13k genes)