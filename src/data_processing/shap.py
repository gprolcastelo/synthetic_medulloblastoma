import os, shap, torch
import numpy as np
from tqdm import tqdm
from src.data_processing.classification import classification_benchmark
import xgboost as xgb
import pandas as pd

def sum_shap_values(shap_values,q_here,cols,idx):
    """Get the sum of SHAP value magnitudes over all samples
    """
    shap_values_summed = np.sum(np.abs(shap_values), axis=0)
    q_shap_values = np.quantile(shap_values_summed, q_here, axis=0)
    tf_shap_values = shap_values_summed > q_shap_values
    important_features = pd.DataFrame(tf_shap_values,columns=cols,index=idx)
    return important_features, shap_values_summed

def xgboost_shap(embeddings, xgb_model):
    print('--> inside xgboost_shap')
    print('xgb_model:', xgb_model)
    print('isinstance(xgb_model, xgb.Booster) =', isinstance(xgb_model, xgb.Booster))

    try:
        # SHAP Tree Explainer to get the importance of the features
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(embeddings)  # we want to explain the whole dataset
    except KeyError as e:
        if str(e) == "'model'":
            print("Skipping SHAP TreeExplainer due to KeyError: 'model'. Returning None as explainer and shap_values.")
            return None, None
        else:
            raise e  # If it's a different KeyError, raise it

    return explainer, shap_values

def vae_shap(data, model_vae,scaler):
    # MinMaxScaler
    # scaler = MinMaxScaler()
    # scaler.fit(data)
    data2explain = scaler.transform(data)
    data2explain = torch.tensor(data2explain).float()
    # SHAP Deep Explainer to get the importance of the features
    deep_explainer = shap.DeepExplainer(model_vae, data2explain)
    deep_shap_values = deep_explainer.shap_values(data2explain)

    return deep_explainer, deep_shap_values

def bagging_shap_pipeline(n_bags, X_data, y_data, z_data,
                          classification_type,
                          num_classes,
                          test_size=0.2, n_br=100,
                          num_threads=os.cpu_count(),
                          n_trials=100, save_path=None):
    print('Starting bagging shap pipeline')
    print('save_path =', save_path)
    print('X_data.shape =', X_data.shape)
    groups = ['Group3', 'Group4', 'SHH', 'WNT']
    bagging_shap_values, metrics, all_params = [], [], []
    # seeds = []
    seeds = np.random.randint(0 ,int(1e9), n_bags).tolist()
    for seed_i in tqdm(seeds , desc='XGBoost Bagging Shap'):
        # seed_i = np.random.randint(0 ,int(1e9))
        # seeds.append(seed_i)

        # 1. Optuna to get optimal parameters for XGBoost
        # 2. XGBoost to obtain predictions from latent space
        print('Starting optuna + xgboost')
        (tree_model, metrics_i, _, _, _, all_params_i) = classification_benchmark(
            z_data, y_data, classification_type,
            seed=seed_i, test_size=test_size ,n_br=n_br,
            num_classes=num_classes ,num_threads=num_threads,
            n_trials=n_trials,
        )
        print('tree_model = ', tree_model)
        metrics.append(metrics_i)
        all_params.append(all_params_i)
        print('Done optuna + xgboost')
        # 3. Tree SHAP in latent space
        print('Starting shap in ls classification')
        explainer, shap_values = xgboost_shap(z_data, tree_model)
        print('xgboost shap_values.shape =', shap_values.shape)
        bagging_shap_values.append(shap_values)
        print('Done shap in ls classification')
        # save results to csv
        if save_path is not None:
            # create save path for the current seed
            save_path_i = os.path.join(save_path, 'tree', str(seed_i))
            print('save_path_i =', save_path_i)
            os.makedirs(save_path_i, exist_ok=True)
            # Save metrics
            metrics_i.to_csv(os.path.join(save_path_i, "classification_metrics.csv"))
            # Save parameters
            pd.DataFrame([all_params_i]).to_csv(os.path.join(save_path_i, "xgboost_parameters.csv"))
            # Transforming 3D array to DataFrame
            ## Flatten the array along axis 0
            flattened_classification = np.reshape(shap_values, newshape=(shap_values.shape[0], -1))
            ## Generate column names
            column_names_classification = [f"lv_{i}_group_{groups[j]}" for i in range(shap_values.shape[1]) for j in
                            range(shap_values.shape[2])]
            ## Convert to DataFrame
            df_classification = pd.DataFrame(flattened_classification, columns=column_names_classification, index=X_data.index)
            ## Save to CSV
            #df_classification.to_csv(os.path.join(save_path_i, "classification_shap_values.csv"))
    return bagging_shap_values, metrics, all_params, seeds

def deep_bagging_shap_pipeline(n_bags, X_data, deep_model, scaler, save_path=None):
    print('Starting deep bagging shap pipeline')
    print('save_path =', save_path)
    print('X_data.shape =', X_data.shape)
    groups = ['Group3', 'Group4', 'SHH', 'WNT']
    deep_bagging_shap_values= []
    seeds = np.random.randint(0, int(1e9), n_bags).tolist()
    for seed_i in tqdm(seeds, desc='Deep Bagging Shap'):
    # for _ in tqdm(range(n_bags) , desc='Deep Bagging Shap'):
    #     seed_i = np.random.randint(0 ,int(1e9))
    #     seeds.append(seed_i)
        # 4. Deep SHAP to map latent space components to genes
        print('Starting shap on vae')
        deep_explainer, deep_shap_values = vae_shap(X_data, deep_model, scaler)
        print('deep_shap_values.shape =', deep_shap_values.shape)
        deep_bagging_shap_values.append(deep_shap_values)
        print('Done shap on vae')
        # save results to csv
        if save_path is not None:
            # create save path for the current seed
            save_path_i = os.path.join(save_path, 'deep',str(seed_i))
            print('save_path_i =', save_path_i)
            os.makedirs(save_path_i, exist_ok=True)
            # Transforming 3D array to DataFrame
            ## Flatten the array along axis 0
            flattened_deep = np.reshape(deep_shap_values, newshape=(deep_shap_values.shape[0], -1))
            ## Generate column names
            column_names_deep = [f"gene_{i}_lv_{j}" for i in tqdm(X_data.columns) for j in range(deep_shap_values.shape[-1])]
            ## Convert to DataFrame
            df_deep = pd.DataFrame(flattened_deep,index=X_data.index, columns=column_names_deep)
            ## Save to CSV
            #df_deep.to_csv(os.path.join(save_path_i, "deep_shap_values.csv"))
    return deep_bagging_shap_values, seeds