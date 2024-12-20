import argparse
import os
from datetime import datetime
today = datetime.today().strftime('%Y%m%d')

# Create the parser
parser = argparse.ArgumentParser(description='Train VAE with given hyperparameters')

# Add arguments
parser.add_argument('--md', type=int, required=True, help='The md value')
parser.add_argument('--f', type=int, required=True, help='The f value')
parser.add_argument('--lr', type=float, required=True, help='The lr value')
parser.add_argument('--path_rnaseq', type=str,
                    default='../data/interim/20240213_data_VAE_ductal/rnaseq.csv',
                    help='The directory of thernaseq data')
parser.add_argument('--path_clinical', type=str,
                    default='../data/interim/20230905_preprocessed_anew/CuratedClinicalData.csv',
                    help='The directory of the clinical metadata')
parser.add_argument('--save_path', type=str, help='The directory to save the model')
parser.add_argument('--save_model', action='store_true',
                    help='Whether to save the model or not')
parser.add_argument('--save_model_path', type=str,help='The directory to save the model in .pth format')
parser.add_argument('--cvae', action='store_true', help='Whether to use CVAE or not')
parser.add_argument('--batch_size', type=int, default=8, help='The batch size')
# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
mid_dim = args.md
features = args.f
lr = args.lr
name=f'retraining_md{mid_dim}_f{features}_lr{lr}'
path_clinical=args.path_clinical
path_rnaseq=args.path_rnaseq
# input_data_type=args.input_data_type
cvae=args.cvae

# Imports:
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Custom functions:
from models.train_model import *



# Parameters:
ch_batch_size = args.batch_size
loss = 'mse'

# Cycle-annealing hyperparameters:
ch_epochs = 200
ch_cycles = 3
ch_width = 80
ch_reduction = 20
ch_beta = 1

# Number of stages of cancer; this will be used by the classifier in VAE_clinical
num_classes = 4
###########################################
# RUN

# Load data as tensor:

# train_data, train_loader, test_data, test_loader = data2tensor(
#     train_path = train_path,
#     test_path = test_path,
#     batch_size = ch_batch_size
# )



if cvae:
    save_path =os.path.join(args.save_path,f'{today}_CVAE_{name}')
    os.makedirs(save_path, exist_ok=True)
    model_type = 'CVAE'
    from src.models.train_model import as_dataloader
    from models.my_model import CVAE
    print('Using CVAE...')
    # Load data
    data_tensor, labels_onehot, loader_index_train, loader_index_test = load_data_cvae(
        path_rnaseq=path_rnaseq,
        path_clinical=path_clinical,
        batch_size=ch_batch_size
    )
    # print('type(data_tensor) =\t',type(data_tensor))
    # print('type(labels_onehot) =\t',type(labels_onehot))
    # Load model
    input_dim = data_tensor.shape[1]
    num_classes = labels_onehot.shape[1]
    # print('input_dim =\t',input_dim)
    # print('num_classes =\t',num_classes)
    chosen_model = CVAE(
        input_dim=input_dim,
        mid_dim=mid_dim,
        features=features,
        num_classes=num_classes,
    )


    # Train cyclically:
    loss, kl, rec, device = cyclical_training_cvae(
        save_path=args.save_model_path,
        model=chosen_model,
        loader_train_idx=loader_index_train,
        loader_test_idx=loader_index_test,
        data=data_tensor,
        labels=labels_onehot,
        epochs=ch_epochs,
        cycles=ch_cycles,
        beta=ch_beta,
        option=loss,
        learning_rate=lr,
        save_model=True
    )
    tr_l, tt_l = loss['train'], loss['test']
    tr_kl, tt_kl = kl['train'], kl['test']
    tr_r, tt_r = rec['train'], rec['test']

else:
    save_path = os.path.join(args.save_path,f'{today}_VAE_{name}')
    os.makedirs(save_path, exist_ok=True)
    model_type = 'VAE'
    print('Using VAE...')
    from models.my_model import VAE


    # Load data as tensor:
    train_data, train_loader, test_data, test_loader, loader_train_clinical, loader_test_clinical = data2tensor(
        path_rnaseq = path_rnaseq,
        path_clinical = path_clinical,
        batch_size = ch_batch_size,
        cvae = cvae,
        wsr = False,
        save_path=save_path,
    )

    input_dim = train_data.shape[0]
    num_patients = train_data.shape[1]+test_data.shape[1]
    print('Data Preprocessing successfully')

    print('input_dim =\t',input_dim)
    print('mid_dim =\t',mid_dim)
    print('features =\t',features)
    print('lr =\t',lr)
    print('num_patients =\t',num_patients)
    print('num_classes =\t',num_classes)
    # Initialize model:
    chosen_model = VAE(
        input_dim=input_dim,
        mid_dim=mid_dim,
        features=features,
    )
    # Create README:
    create_readme(
        today, path_rnaseq, path_clinical, save_path,
        ch_batch_size, input_dim, mid_dim, features,
        lr, loss, ch_epochs, ch_cycles, ch_width,
        ch_reduction, ch_beta, model_type,
        "standard"
    )
    (tr_l, tt_l, tr_kl, tt_kl, tr_r, tt_r, tt_dev) = cyclical_training(
        save_path=args.save_model_path,
        model=chosen_model,
        loader_train=train_loader,
        loader_test=test_loader,
        epochs=ch_epochs,
        cycles=ch_cycles,
        initial_width=ch_width,
        reduction=ch_reduction,
        beta=ch_beta,
        option=loss,
        learning_rate=lr,
        class_data_train=loader_train_clinical,
        class_data_test=loader_test_clinical,
        save_model=args.save_model,
        model_type=model_type,
        cvae=cvae,
    )

# Draw and save loss plots:
loss_plots(
    save_path=save_path,
    train_loss=tr_l,
    test_loss=tt_l,
    kl_loss_train=tr_kl,
    kl_loss_test=tt_kl,
    rec_loss_train=tr_r,
    rec_loss_test=tt_r,
)


# Use in command line. For example:
# Train VAE on ductal deseq data:
# python python_VAE.py --md 512 --f 32 --lr 0.0001 --train_test_dir ../data/interim/20240125_train_test_deseq_ductal/ --save_model False --model_type VAE --num_flows 0 --input_data_type deseq
# Train VAE on ductal standard data, saving the model:
# python python_VAE.py --md 1024 --f 32 --lr 0.00001 --train_test_dir ../data/interim/20240123_train_test_ductal/ --save_model True --model_type VAE --num_flows 0 --input_data_type standard

# python python_VAE.py --md 1024 --f 32 --lr 0.0001 --train_test_dir ../data/interim/20240123_train_test_ductal/ --save_model False --input_data_type standard