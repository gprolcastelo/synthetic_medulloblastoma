import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from datetime import datetime
from sklearn.model_selection import train_test_split

global torch_dtype
torch_dtype = torch.float32
torch.set_default_dtype(torch_dtype)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds:

def set_seed(seed=2023):
    '''
    This function sets the seed for various random number generators used in deep learning frameworks like PyTorch, NumPy, and Caffe2. 
    Providing a fixed seed ensures consistent results across multiple runs, which is crucial for training and evaluating machine learning models.

    Parameters:
        seed (int, optional): The random seed to set. Defaults to 2023.
    '''

    # Set the random number generator seed
    random.seed(seed)  

    # Set the NumPy random number seed
    np.random.seed(seed) 

    # Set the PyTorch seed
    if torch.cuda.is_available():
        # Set the CUDA seed
        torch.cuda.manual_seed_all(seed) 

    # Set the deterministic flag for CuDNN
    torch.backends.cudnn.deterministic = True 

    # Disable CuDNN's benchmark mode
    torch.backends.cudnn.benchmark = False 

def create_readme(
    today, train_path, test_path, save_path,
    ch_batch_size, input_dim, mid_dim, features,
    lr, loss, ch_epochs, ch_cycles, ch_width,
    ch_reduction, ch_beta, model_type, #num_flows,
    input_data_type
):
    '''
    Create README.md file with experiment details.

    Parameters:
    - today (str): Date of the experiment in the format 'YYYYMMDD'.
    - train_path (str): Path to the training data.
    - test_path (str): Path to the testing data.
    - save_path (str): Path to save the experiment results.
    - ch_batch_size (int): Batch size for training.
    - input_dim (int): Input dimension of the model.
    - mid_dim (int): Mid dimension of the model.
    - features (int): Number of features in the model.
    - lr (float): Learning rate for training.
    - loss (str): Loss function used for training.
    - ch_epochs (int): Number of epochs for training.
    - ch_cycles (int): Number of cycles for training.
    - ch_width (int): Width parameter for the model.
    - ch_reduction (float): Reduction parameter for the model.
    - ch_beta (float): Beta parameter for the model.
    '''

    # Remove README if it already exists:
    if os.path.isfile(os.path.join(save_path, 'README.md')):
        os.remove(os.path.join(save_path, 'README.md'))


    # Create README content
    readme_content = f'''# Experiment {today}:
## Paths:
* train_path = {train_path}
* test_path = {test_path}
* save_path = {save_path}
## Model:
* model_type = {model_type}
## Parameters:
* input_data_type = {input_data_type}
* ch_batch_size = {ch_batch_size}
* input_dim = {input_dim}
* mid_dim = {mid_dim}
* features = {features}
* lr = {lr}
* loss = {loss}
* ch_epochs = {ch_epochs}
* ch_cycles = {ch_cycles}
* ch_width = {ch_width}
* ch_reduction = {ch_reduction}
* ch_beta = {ch_beta}'''

    # Write content to README file
    readme_path = f"{save_path}/README.md"
    with open(readme_path, "w") as readme_file:
        readme_file.write(readme_content)

def setup(model, learning_rate=0.0001, option='mse'):
    '''
    Setups the necessary components for training a neural network, including the learning rate, optimizer, criterion, and device.
    
    Parameters:
    model (nn.Module): The neural network model to be trained.
    learning_rate (float, optional): The initial learning rate for the optimizer. Defaults to 0.0001.
    option (str, optional): A string that determines the type of loss function to use. Can be either 'bce' for binary cross-entropy loss or 'mse' for mean squared error loss. Defaults to 'mse'.
    
    Returns:
    optimizer (Optimizer): An instance of the Adam optimizer.
    criterion (nn.Criterion): An instance of the BCELoss or MSELoss criterion.
    device (torch.device): The chosen device ("cuda" if torch.cuda.is_available() else "cpu").
    '''
    # Create the criterion
    if option == 'bce':
        criterion = nn.BCELoss(reduction='sum')  # Use binary cross-entropy loss for bce option
    else:
        criterion = nn.MSELoss(reduction='sum')  # Use mean squared error loss for mse option
    
    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Create an instance of the Adam optimizer
    
    # Choose the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose cuda device if available, cpu otherwise
    
    return optimizer, criterion, device  # Return the optimizer, criterion, and device
    
# Final loss function with beta as hyperparameters
def calculate_loss(mu, logvar, reconstruction_loss, beta, bce_loss=None):
    '''
    Computes the final loss function with beta as a hyperparameter.
    
    Parameters:
    mu (tensor): Mean of the approximate posterior distribution.
    logvar (tensor): Logarithmic variance of the approximate posterior distribution.
    reconstruction_loss (tensor): Reconstruction loss.
    beta (float): Hyperparameter controlling the strength of the KL divergence term.
    
    Returns:
    loss (tensor): Total loss.
    kl_div (tensor): KL divergence term.
    '''
    # Compute KL divergence term
    kl_div = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    
    # Compute reconstruction loss
    #reconstruction = reconstruction_loss
    
    # Compute total loss
    loss = beta * kl_div + reconstruction_loss
    # Kingma 2014, Semi-Supervised Learning with Deep Generative Models, eq. 9:
    # Multiply classification loss by parameter alpha:
    # alpha = 0.1*data_train.shape[1] (i.e., number of samples)
    if bce_loss is not None:
        loss += bce_loss
    return loss, kl_div



# Fitting function
def fit(model, dataloader, beta, optimizer, criterion, device,cvae=False,class_data=None):
    '''
    Trains a neural network model for one epoch and computes the total loss, KL divergence, and reconstruction loss.

    Parameters:
    model (nn.Module): The neural network model to be trained.
    dataloader (DataLoader): DataLoader for the training data.
    beta (float): Hyperparameter controlling the strength of the KL divergence term.
    optimizer (Optimizer): An instance of the Adam optimizer.
    criterion (nn.Criterion): An instance of the BCELoss or MSELoss criterion.
    device (torch.device): The chosen device ("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
    train_loss (float): The average training loss for the epoch.
    kl_div_loss (float): The average KL divergence for the epoch.
    reconstructed_loss (float): The average reconstruction loss for the epoch.
    '''
    model.train()
    running_loss:float = 0.0
    rec_loss: float = 0.0
    kl_loss: float = 0.0

    if cvae:
        for data, class_data_item in zip(dataloader,class_data):
            data = data.to(device)
            class_data_item = class_data_item.to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar, z = model(data, class_data_item)
            # Compute reconstruction loss:
            reconstruction_loss = criterion(reconstruction, data)
            loss, kl = calculate_loss(mu, logvar, reconstruction_loss, beta, bce_loss=None)
            # Add running losses:
            rec_loss += reconstruction_loss.item()
            kl_loss += kl.item()
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
    else:
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar, z = model(data)
            # Compute reconstruction loss:
            reconstruction_loss = criterion(reconstruction, data)
            loss, kl = calculate_loss(mu, logvar, reconstruction_loss, beta)
            # Add running losses:
            rec_loss += reconstruction_loss.item()
            kl_loss += kl.item()
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
    # Save profiling results
    # Average losses:
    train_loss = running_loss/len(dataloader.dataset)
    kl_div_loss = kl_loss/len(dataloader.dataset)
    reconstructed_loss = rec_loss/len(dataloader.dataset)

    return train_loss, kl_div_loss, reconstructed_loss, None
    
# Validation over test dataset
def validate(model, dataloader, beta, criterion, device,cvae=False,class_data=None,num_patients=0,model_type='VAE'):
    '''
    Evaluates a neural network model on a validation set and computes
    the total loss, KL divergence, and reconstruction loss.

    Parameters:
    model (nn.Module): The neural network model to be evaluated.
    dataloader (DataLoader): DataLoader for the validation data.
    beta (float): Hyperparameter controlling the strength of the KL divergence term.
    criterion (nn.Criterion): An instance of the BCELoss or MSELoss criterion.
    device (torch.device): The chosen device ("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
    val_loss (float): The average validation loss.
    kl_div_loss (float): The average KL divergence for the validation set.
    reconstructed_loss (float): The average reconstruction loss for the validation set.
    '''
    model.eval()  # network in evaluation mode
    running_loss = 0.0
    rec_loss = 0.0
    kl_loss = 0.0

    with torch.no_grad():  # in validation we don't want to update weights
        if cvae:
            for data, class_data_item in zip(dataloader,class_data):
                # print('Not a warm-up cycle validation')
                data = data.to(device)
                class_data_item = class_data_item.to(device)
                reconstruction, mu, logvar, z  = model(data, class_data_item)
                # Compute reconstruction loss:
                reconstruction_loss = criterion(reconstruction, data)
                loss, kl = calculate_loss(mu, logvar, reconstruction_loss, beta, bce_loss=None)
                # Add running losses:
                rec_loss += reconstruction_loss.item()
                kl_loss += kl.item()
                running_loss += loss.item()
        else:
            print('Validating VAE')
            for data in dataloader:
                data = data.to(device)
                reconstruction, mu, logvar, z = model(data)
                # Compute reconstruction loss:
                reconstruction_loss = criterion(reconstruction, data)
                loss, kl = calculate_loss(mu, logvar, reconstruction_loss, beta)

                # Add running losses:
                running_loss += loss.item()
                kl_loss += kl.item()
                rec_loss += reconstruction_loss.item()

    val_loss = running_loss / len(dataloader.dataset)
    kl_div_loss = kl_loss / len(dataloader.dataset)
    reconstructed_loss = rec_loss / len(dataloader.dataset)
    return val_loss, kl_div_loss, reconstructed_loss, None

def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    '''
    Generates a linear schedule for the beta hyperparameter over a specified number of epochs and cycles.

    Parameters:
    start (float): The initial value of the hyperparameter.
    stop (float): The final value of the hyperparameter.
    n_epoch (int): The total number of epochs.
    n_cycle (int, optional): The number of cycles. Defaults to 4.
    ratio (float, optional): The ratio of the cycle length that is increasing. Defaults to 0.5.

    Returns:
    L (numpy array): An array of length n_epoch with the scheduled hyperparameter values.
    '''
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L



# Defined cyclical training
def cyclical_training(save_path, model, loader_train, loader_test,
                      epochs=110, cycles=3, initial_width=80, reduction=20,
                      beta=1.00, option="mse", learning_rate=0.0001,save_model=False,
                      cvae=False, class_data_train=None, class_data_test=None,model_type='VAE'):
    '''
    Performs cyclical training of a neural network model. Saves model if specified.

    Parameters:
    model (nn.Module): The neural network model to be trained.
    save_path (str): Path to save the training results.
    loader_train (DataLoader): DataLoader for the training data.
    loader_test (DataLoader): DataLoader for the testing data.
    epochs (int, optional): The total number of epochs for each cycle. Defaults to 110.
    cycles (int, optional): The number of cycles. Defaults to 3.
    initial_width (int, optional): The initial width for the model. Defaults to 80.
    reduction (int, optional): The reduction parameter for the model. Defaults to 20.
    beta (float, optional): Hyperparameter controlling the strength of the KL divergence term. Defaults to 1.00.
    option (str, optional): A string that determines the type of loss function to use. Can be either 'bce' for binary cross-entropy loss or 'mse' for mean squared error loss. Defaults to 'mse'.
    learning_rate (float, optional): The initial learning rate for the optimizer. Defaults to 0.0001.
    save_model (bool, optional): Whether to save the trained model. Defaults to False.

    Returns:
    train_loss (list): List of average training losses for each epoch.
    test_loss (list): List of average testing losses for each epoch.
    kl_loss_train (list): List of average KL divergence losses for the training set for each epoch.
    kl_loss_test (list): List of average KL divergence losses for the testing set for each epoch.
    rec_loss_train (list): List of average reconstruction losses for the training set for each epoch.
    rec_loss_test (list): List of average reconstruction losses for the testing set for each epoch.
    time_today (str): The current time in the format 'HHMMSS'.
    device (torch.device): The chosen device ("cuda" if torch.cuda.is_available() else "cpu").
    '''
    # Number of patients (samples) in total:
    num_patients = len(loader_train.dataset) + len(loader_test.dataset)
    # Get time and date with zeros before single digits:
    today = datetime.today().strftime('%Y%m%d')

    os.makedirs(save_path,exist_ok=True)
    
    train_loss = []
    test_loss = []
    kl_loss_train = []
    kl_loss_test = []
    rec_loss_train = []
    rec_loss_test = []
    # if clinical_train:
    #     bce_train_loss_collect = []
    #     bce_test_loss_collect = []
    # else:
    #     bce_train_loss_collect = None
    #     bce_test_loss_collect = None

    
    # Setup
    set_seed()
    optimizer, criterion, device = setup(model, learning_rate, option)
    model = model.to(device)
    beta_lu = frange_cycle_linear(0.0, beta, epochs*cycles, cycles, ratio=0.5)

    # Training cycles
    ind = 0
    global cycle
    for cycle in range(cycles):

        print("Starting Cycle ", cycle)
        for epoch in tqdm(range(epochs)):

            # get beta for cycle annealing
            beta_launcher = beta_lu[int(ind)]
            # go to the next index of beta_lu
            ind+=1
            # fit model with training data
            train_epoch_loss, kl_train_loss, rec_train_loss, bce_train_loss = fit(model, loader_train, beta_launcher,
                                                                                  optimizer, criterion, device,
                                                                                  cvae, class_data_train)

            # validate model with test data
            test_epoch_loss, kl_test_loss, rec_test_loss, bce_test_loss = validate(model, loader_test, beta_launcher,
                                                                                   criterion, device,
                                                                                   cvae, class_data_test)
            # Append losses to lists
            train_loss.append(train_epoch_loss)
            test_loss.append(test_epoch_loss)
            kl_loss_train.append(kl_train_loss)
            kl_loss_test.append(kl_test_loss)
            rec_loss_train.append(rec_train_loss)
            rec_loss_test.append(rec_test_loss)
            # if clinical_train:
            #     bce_train_loss_collect.append(bce_train_loss)
            #     bce_test_loss_collect.append(bce_test_loss)
    #####################################################        
    #                Save the model trained             #
    #####################################################

    # if save_model:
    #     PATH = "../models/" + today + "_VAE_idim" + str(model.input_dim) + "_md" + str(
    #         model.mid_dim) + "_feat" + str(model.features) + option + "_relu.pth"
    #     print("Trained VAE model saved as " + PATH)
    #     torch.save(model.state_dict(), PATH)  # save in a dictionary all parameters
    # else:
    #     print("Model not saved")

    if save_model:
        # Check model.mid_dim exists:
        if hasattr(model, 'mid_dim'):
            PATH = os.path.join(
            save_path,
            today + f"_{model_type}_idim" + str(model.input_dim) + "_md" + str(model.mid_dim) + "_feat" + str(model.features) + "_lr" + str(learning_rate) + ".pth"
            )
        else:
            PATH = os.path.join(
            save_path,
            today + f"_{model_type}3layer_idim" + str(model.input_dim) + "_feat" + str(model.features) + ".pth"
            )
        os.makedirs(save_path,exist_ok=True)
        print("Trained VAE model saved as " + PATH)
        torch.save(model.state_dict(), PATH)  # save in a dictionary all parameters
    else:
        print("Model not saved")

    return train_loss, test_loss, kl_loss_train, kl_loss_test, rec_loss_train, rec_loss_test, device #bce_train_loss_collect, bce_test_loss_collect, device

def cyclical_training_cvae(
        save_path, model, loader_train_idx, loader_test_idx,
        data, labels,
        epochs=110, cycles=3,
        beta=1.00, option="mse", learning_rate=0.0001,
        save_model=False
):
    today = datetime.today().strftime('%Y%m%d')
    os.makedirs(save_path,exist_ok=True)
    # Setup
    set_seed()
    optimizer, criterion, device = setup(model, learning_rate, option)
    model = model.to(device)
    beta_lu = frange_cycle_linear(0.0, beta, epochs*cycles, cycles, ratio=0.5)
    # Create dictionaries to store losses
    loss={'train': [], 'test': []}
    kl={'train': [], 'test': []}
    rec={'train': [], 'test': []}

    # Training cycles
    ind = 0
    global cycle
    for cycle in range(cycles):
        print("Starting Cycle ", cycle)
        for epoch in tqdm(range(epochs)):
            # get beta for cycle annealing
            beta_launcher = beta_lu[int(ind)]
            # go to the next index of beta_lu
            ind+=1
            # fit model with training data
            train_epoch_loss, kl_train_loss, rec_train_loss = fit_cvae(
                model=model,
                dataloader=loader_train_idx,
                data=data,
                labels=labels,
                beta=beta_launcher,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            # Append losses to lists
            loss['train'].append(train_epoch_loss)
            kl['train'].append(kl_train_loss)
            rec['train'].append(rec_train_loss)

            # validate model with test data
            test_epoch_loss, kl_test_loss, rec_test_loss = validate_cvae(
                model=model,
                dataloader=loader_test_idx,
                data=data,
                labels=labels,
                beta=beta_launcher,
                criterion=criterion,
                device=device,
            )
            # Append losses to lists
            loss['test'].append(test_epoch_loss)
            kl['test'].append(kl_test_loss)
            rec['test'].append(rec_test_loss)

    # Save model
    if save_model:
        # Check there is a folder in the current directory called 'models':
        if not os.path.exists('models'):
            os.makedirs('models')
        save_path = 'models'
        # print('Saving model to:',save_path)
        save_path = os.path.join(save_path, f"{today}_CVAE")
        os.makedirs(save_path, exist_ok=True)
        PATH = os.path.join(
            save_path,
            today + f"_CVAE_idim" + str(model.input_dim) + "_md" + str(model.mid_dim) + "_feat" + str(
                model.features) + option + "_relu.pth"
        )
        print('Saving model to:',PATH)
        torch.save(model.state_dict(), PATH)  # save in a dictionary all parameters

    return loss, kl, rec, device

def fit_cvae(model, dataloader, data, labels, beta, optimizer, criterion, device):
    """
    This function takes in the dataloader of the indices to be used for training, instead of the data itself.
    The index of the dataloader is used to index the data and class_data to be used for training.
    This way, problems with shuffling data and class_data with dataloaders are avoided.
    :param model:
    :param dataloader:
    :param data:
    :param labels:
    :param beta:
    :param optimizer:
    :param criterion:
    :param device:
    :return:
    """
    model.train()
    running_loss: float = 0.0
    rec_loss: float = 0.0
    kl_loss: float = 0.0
    for i in dataloader:
        # print('i =\t', i)
        # print('i.shape =\t', i.shape)
        x = data[i].to(device)
        # print('x.shape =\t', x.shape)
        c = labels[i].to(device)
        # print('c.shape =\t', c.shape)
        optimizer.zero_grad()
        reconstruction, mu, logvar, z = model(x, c)
        # print('reconstruction.shape =\t', reconstruction.shape)
        # Compute reconstruction loss:
        reconstruction_loss = criterion(reconstruction, x)
        loss, kl = calculate_loss(mu, logvar, reconstruction_loss, beta)
        # Add running losses:
        rec_loss += reconstruction_loss.item()
        kl_loss += kl.item()
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    # Average losses:
    train_loss = running_loss / len(dataloader.dataset)
    kl_div_loss = kl_loss / len(dataloader.dataset)
    reconstructed_loss = rec_loss / len(dataloader.dataset)
    return train_loss, kl_div_loss, reconstructed_loss

def validate_cvae(model, dataloader, data, labels, beta, criterion, device):
    """
    This function takes in the dataloader of the indices to be used for training, instead of the data itself.
    The index of the dataloader is used to index the data and class_data to be used for training.
    This way, problems with shuffling data and class_data with dataloaders are avoided.
    :param model:
    :param dataloader:
    :param data:
    :param labels:
    :param beta:
    :param criterion:
    :param device:
    :return:
    """
    model.eval() # model in evaluation mode
    running_loss: float = 0.0
    rec_loss: float = 0.0
    kl_loss: float = 0.0
    for i in dataloader:
        x = data[i].to(device)
        c = labels[i].to(device)
        reconstruction, mu, logvar, z = model(x, c)
        # Compute reconstruction loss:
        reconstruction_loss = criterion(reconstruction, x)
        loss, kl = calculate_loss(mu, logvar, reconstruction_loss, beta)
        # Add running losses:
        rec_loss += reconstruction_loss.item()
        kl_loss += kl.item()
        running_loss += loss.item()
    # Average losses:
    val_loss = running_loss / len(dataloader.dataset)
    kl_div_loss = kl_loss / len(dataloader.dataset)
    reconstructed_loss = rec_loss / len(dataloader.dataset)
    # print('Average losses')
    # print('val_loss:',val_loss)
    # print('kl_div_loss:',kl_div_loss)
    # print('reconstructed_loss:',reconstructed_loss)
    return val_loss, kl_div_loss, reconstructed_loss

# Global variables for the scalers:
scaler1 = None
scaler2 = None

def normalize_data(data_train, data_test):
    '''
    Parameters:
    Normalizes input data using MinMaxScaler.
    data_train: (pd.DataFrame) Training data.
    data_test: (pd.DataFrame) Testing data.

    Returns:
    normalized_data_train: (pd.DataFrame) Normalized training data.
    normalized_data_test: (pd.DataFrame) Normalized testing data.

    '''
    global scaler1, scaler2
    # MinMax Scaler: values in range [0,1]
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()

    # Fit the scaler1 with the training data
    scaler1.fit(data_train)
    normalized_data_train = scaler1.transform(data_train)

    # Fit the scaler2 with the fitted scaler1
    scaler2.fit(data_train)
    scaler2.fit(data_test)
    # Use the same scaler2 to transform the test data
    normalized_data_test = scaler2.transform(data_test)

    return normalized_data_train, normalized_data_test


def split_data(path_rnaseq, path_clinical,cvae=False, test_size=0.2,seed=2023):
    '''
    Splits the data into training and testing sets using stratified sampling.

    Parameters:
    path_rnaseq (str): The path to the RNA-seq data.
    path_clinical (str): The path to the clinical metadata.
    test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
    seed (int, optional): The random seed. Defaults to 2023.

    Returns:
    train_data (DataFrame): The training data.
    test_data (DataFrame): The testing data.
    '''
    # Set the random seed
    set_seed(seed)
    # Load the data
    # rnaseq = pd.read_csv(path_rnaseq,index_col=0)
    rnaseq = pd.read_csv(path_rnaseq,index_col=0)
    clinical = pd.read_csv(path_clinical, index_col=0)
    # # Check number of samples in rnaseq and clinical data:
    if rnaseq.shape[0] != clinical.shape[0]:
        rnaseq = rnaseq.T
    assert rnaseq.shape[0] == clinical.shape[0], "Number of samples in RNA-seq and clinical data do not match."
    # Get ductal patients:
    # clinical = clinical[clinical["histological_type"] == 'Infiltrating Ductal Carcinoma']
    # Filter rnaseq on ductal patients:
    # rnaseq = rnaseq[rnaseq.index.isin(clinical.index)]
    # Split the data into training and testing sets
    y = clinical["Sample_characteristics_ch1"].values.tolist()
    if cvae:
        y_stage = clinical["Sample_characteristics_ch1"].values.tolist()
        # y_type = clinical["histological_type"].values.tolist()
        print('One-hot encoding clinical data')
        # Create a dictionary that maps the groups to integers
        stage_to_int = {'WNT': 1, 'SHH': 2, 'Group3': 3, 'Group4': 4, 'Group 3': 3, 'Group 4': 4}
        # Map cancer type to simplified types:
        # type_conversion = {"Infiltrating Ductal Carcinoma": "ductal",
        #                    "Infiltrating Lobular Carcinoma": "lobular"}

        # Use a list comprehension to replace each group with its integer value
        # print('y_stage=\n',y_stage)
        y_stage = [stage_to_int[subgroup] for subgroup in y_stage]
        # clinical_data_train = [stage_to_int[stage] for stage in clinical_data_train]
        # clinical_data_test = [stage_to_int[stage] for stage in clinical_data_test]
        # Replace histological type with simplified types:
        # y_type = [type_conversion[ht] for ht in y_type]
        # To stratify on both stage and type, we can concatenate the two lists:
        y = y_stage
        # y = [tp + "_" + stg for stg, tp in zip(y_stage, y_type)]
        categs = sorted(set(y))
        # print('categs=\n', categs)
        print("The shape of the clinical train data is (patients: samples): ", len(y))
        # One-hot encode the clinical data for classification:
        y = np.array(y).reshape(-1, 1) # reshape for single feature


        # print('before ohe: y[:10]=\n',y[:10])
        # For torch version 1, call attribute sparse:
        ohe = OneHotEncoder(categories=[categs], handle_unknown='ignore', sparse_output=False, dtype=np.int8).fit(y)
        y = ohe.transform(y)
        # print('y.shape after ohe =\t',y.shape)
        # print('y[:10,:]=\n',y[:10,:])

    X_train, X_test, y_train, y_test = train_test_split(rnaseq, y, test_size=test_size, stratify=y)
    # Final shape must be (samples, features):
    # X_train = X_train.T
    # X_test = X_test.T
    return X_train, X_test, y_train, y_test



def data2tensor(path_rnaseq, path_clinical, batch_size, cvae=False,wsr=False,save_path='../data/interim/'):
    '''
    Loads data from CSV files, normalizes it using MinMaxScaler,
    converts it to tensors, and creates DataLoaders.

    Parameters:
    path_rnaseq (str): The path to the RNA-seq data.
    path_clinical (str): The path to the clinical metadata.
    batch_size (int): The size of the batches for the DataLoader.

    Returns:
    train_dataset (tensor): Tensor of the training data.
    loader_train (DataLoader): DataLoader for the training data.
    test_dataset (tensor): Tensor of the testing data.
    loader_test (DataLoader): DataLoader for the testing data.
    '''
    # Split the data into training and testing sets:
    data_train, data_test, clinical_data_train, clinical_data_test  = split_data(path_rnaseq,
                                                                                 path_clinical,
                                                                                 cvae=cvae,
                                                                                 test_size=0.2,
                                                                                 seed=2023
                                                                                 )
    # print("The shape of the train data is (patients: samples, genes: features): ", data_train.shape)
    # print("The shape of the test data is (patients: samples, genes: features): ", data_test.shape)
    # Kingma 2014, Semi-Supervised Learning with Deep Generative Models, eq. 9:
    # Multiply classification loss by parameter alpha:
    # alpha = 0.1*data_train.shape[1]
    # Data will be transposed for training:
    data_train = data_train.T
    data_test = data_test.T
    print("The shape of the train data is (genes: features, patients: samples): ", data_train.shape)
    print("The shape of the test data is (genes: features, patients: samples): ", data_test.shape)
    print("Data will be transposed when calling DataLoader to get (samples, features) shape")
    # Normalize data:
    normalized_data_train, normalized_data_test = normalize_data(data_train, data_test)
    normalized_data_train = pd.DataFrame(normalized_data_train)
    normalized_data_test = pd.DataFrame(normalized_data_test)
    train_dataset = torch.tensor(normalized_data_train.values).to(torch_dtype)
    test_dataset = torch.tensor(normalized_data_test.values).to(torch_dtype)
    # Create a DataLoader for the training data
    loader_train = torch.utils.data.DataLoader(
        train_dataset.T,
        batch_size=batch_size,
        shuffle=True,
    )
    # Create a DataLoader for the testing data
    loader_test = torch.utils.data.DataLoader(
        test_dataset.T,
        batch_size=batch_size,
        shuffle=False,
    )
    print('DataLoader created')
    print('len(loader_test.dataset)=',len(loader_test.dataset))
    print('DataLoader "normal" created')
    print('len(loader_test.dataset)=', len(loader_test.dataset))
    print('len(loader_train.dataset)=', len(loader_train.dataset))
    print('loader_test=\t',loader_test)
    print('loader_train=\t',loader_train)
    print('loader_train.dataset.shape=\t',loader_train.dataset.shape)
    print('loader_test.dataset.shape=\t',loader_test.dataset.shape)
    if cvae:
        # Create a DataLoader for the clinical data, for semi-supervised learning:
        y_train = torch.tensor(clinical_data_train).to(torch_dtype)
        y_test = torch.tensor(clinical_data_test).to(torch_dtype)
        print('y_train.shape=',y_train.shape)
        print('y_test.shape=',y_test.shape)
        loader_train_clinical = torch.utils.data.DataLoader(
            y_train,
            batch_size=batch_size,
            shuffle=True,
        )
        loader_test_clinical = torch.utils.data.DataLoader(
            y_test,
            batch_size=batch_size,
            shuffle=False,
        )

        return train_dataset, loader_train, test_dataset, loader_test, loader_train_clinical, loader_test_clinical
    else:
        return train_dataset, loader_train, test_dataset, loader_test, None, None


def loss_plots(save_path, train_loss, test_loss, kl_loss_train, kl_loss_test, rec_loss_train, rec_loss_test, bce_loss_train=None,bce_loss_test=None):#, time_today):
    '''
    Plots and saves the training and testing losses, KL divergence losses, and reconstruction losses.
    Also saves these losses as CSV files.

    Parameters:
    save_path (str): Path to save the plots and CSV files.
    train_loss (list): List of average training losses for each epoch.
    test_loss (list): List of average testing losses for each epoch.
    kl_loss_train (list): List of average KL divergence losses for the training set for each epoch.
    kl_loss_test (list): List of average KL divergence losses for the testing set for each epoch.
    rec_loss_train (list): List of average reconstruction losses for the training set for each epoch.
    rec_loss_test (list): List of average reconstruction losses for the testing set for each epoch.
    time_today (str): The current time in the format 'HHMMSS'.
    '''
    plt.style.use('ggplot')
    # using now() to get current time:
    time_today = datetime.today().strftime('%H%M%S')
    
    ###########################
    # Loss
    ###########################

    # Linspace for plot and csv:
    x_train_loss = np.linspace(1, len(train_loss), len(train_loss))
    x_test_loss = np.linspace(1, len(test_loss), len(test_loss))

    # Save dataframe as csv:
    list_of_tuples = list(zip(train_loss,test_loss))
    df_loss = pd.DataFrame(list_of_tuples, columns=['train_loss', 'test_loss'])
    df_loss.to_csv(os.path.join(save_path, "loss.csv"))

    # Plot loss
    f = plt.figure(1)
    plt.title('Train Loss vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x_train_loss, train_loss, label = "train")
    plt.plot(x_test_loss, test_loss, label = "test")
    plt.legend()
    
    
    f.savefig(os.path.join(save_path, "Train_VS_Test.png"))

    ###########################
    # Rec Loss
    ###########################

    # Linspace for plot and csv:
    x_train_recloss = np.linspace(1, len(rec_loss_train), len(rec_loss_train))
    x_test_recloss = np.linspace(1, len(rec_loss_test), len(rec_loss_test))

    # Save dataframe as csv:
    list_of_tuples = list(zip(rec_loss_train,rec_loss_test))
    df_loss = pd.DataFrame(list_of_tuples, columns=['rec_loss_train', 'rec_loss_test'])
    df_loss.to_csv(os.path.join(save_path, "rec_loss.csv"))

    # Plot loss

    g = plt.figure(2)
    plt.title('Rec Train Loss vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x_train_recloss, rec_loss_train, label = "rec train")
    plt.plot(x_test_recloss, rec_loss_test, label = "rec test")
    plt.legend()
    g.savefig(os.path.join(save_path, "Rec_Train_VS_Test.png"))

    ###########################
    # KL Loss
    ###########################

    # Linspace for plot and csv:
    x_train_klloss = np.linspace(1, len(kl_loss_train), len(kl_loss_train))
    x_test_klloss = np.linspace(1, len(kl_loss_test), len(kl_loss_test))

    # Save dataframe as csv:
    list_of_tuples = list(zip(kl_loss_train,kl_loss_test))
    df_loss = pd.DataFrame(list_of_tuples, columns=['kl_loss_train', 'kl_loss_test'])
    df_loss.to_csv(os.path.join(save_path, "klloss.csv"))

    # Plot loss

    h = plt.figure(3)
    plt.title('KL Train Loss vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x_train_klloss, kl_loss_train, label = "kl train")
    plt.plot(x_test_klloss, kl_loss_test, label = "kl test")
    plt.legend()
    h.savefig(os.path.join(save_path, "KL_Train_VS_Test.png"))

    ###########################
    # BCE Loss
    ###########################
    if bce_loss_train is not None and bce_loss_test is not None:
        # Linspace for plot and csv:
        x_train_bceloss = np.linspace(1, len(bce_loss_train), len(bce_loss_train))
        x_test_bceloss = np.linspace(1, len(bce_loss_test), len(bce_loss_test))
        # Save dataframe as csv:
        list_of_tuples = list(zip(bce_loss_train,bce_loss_test))
        df_loss = pd.DataFrame(list_of_tuples, columns=['bce_loss_train', 'bce_loss_test'])
        df_loss.to_csv(os.path.join(save_path, "bceloss.csv"))
        # Plot loss
        i = plt.figure(4)
        plt.title('BCE Train Loss vs Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(x_train_bceloss, bce_loss_train, label = "bce train")
        plt.plot(x_test_bceloss, bce_loss_test, label = "bce test")
        plt.legend()
        i.savefig(os.path.join(save_path, "BCE_Train_VS_Test.png"))


def load_data_cvae(path_rnaseq, path_clinical, batch_size):
    # Load data
    data = pd.read_csv(path_rnaseq, index_col=0)
    c = pd.read_csv(path_clinical, index_col=0)
    if data.shape[0] != c.shape[0]:
        data = data.T
    assert data.shape[0] == c.shape[0], "Number of samples in RNA-seq and clinical data do not match."
    # MinMax Scaler: values in range [0,1]
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.values)
    # One-hot encode all the labels
    labels_onehot = pd.get_dummies(c).values
    labels_onehot_tensor = torch.tensor(labels_onehot).float()
    # Data and labels as tensors
    data_tensor = torch.tensor(data_normalized).float()
    # labels_onehot = torch.tensor(labels_onehot).float()
    print('data_flat_tensor.shape =\t', data_tensor.shape)
    print('labels_onehot.shape =\t', labels_onehot.shape)
    # Get a list of indices
    indices = list(range(len(labels_onehot)))
    print('len(indices) =\t', len(indices))
    # Split the indices into training and testing sets with stratification of the condition
    indices_train, indices_test, _, _ = train_test_split(
        indices, labels_onehot, test_size=0.2, random_state=42, stratify=labels_onehot)
    print('len(indices_train) =\t', len(indices_train))
    print('len(indices_test) =\t', len(indices_test))
    # Get indices as dataloaders:
    loader_index_train, loader_index_test = as_dataloader(indices_train, indices_test, batch_size)
    return data_tensor, labels_onehot_tensor, loader_index_train, loader_index_test

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