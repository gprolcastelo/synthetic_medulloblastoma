# Imports
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score, log_loss
import os
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb
from sklearn.preprocessing import label_binarize, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

def classification_metrics(y_test_le, y_pred, num_classes, weights_test=None):
    '''
    Calculate and plot classification metrics: precision, recall, f1-score, AUC-ROC, AUC-PR, balanced accuracy, and Cohen's Kappa.
    :param y_test_le: Actual class labels
    :param y_pred: Predicted probabilities (XGBoost output of model.predict())
    :param weights_test: Weights for the test set. If None, no weights are used (default=None)
    :return: DataFrame with classification metrics, one per column
    '''
    # print('y_pred =', y_pred)
    # Binarize test labels for metrics that need it
    if num_classes > 2:
        y_test_bin = label_binarize(y_test_le, classes=np.arange(num_classes))
        # Get class labels from probabilities
        y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
    elif num_classes == 2:
        y_test_bin = label_binarize(y_test_le, classes=[0, 1])
        y_pred = y_pred[:, 1]  # Select the column for the positive class
        # Get class labels from probabilities
        y_pred_labels = np.where(y_pred > 0.5, 1, 0)  # Convert probabilities to class labels
    else:
        raise ValueError("Number of classes must be greater than 2")

    # Get classification report
    report = classification_report(y_true=y_test_le, y_pred=y_pred_labels, output_dict=True)

    # Get precision, recall, and f1-score for each class
    precision = [report[str(i)]['precision'] for i in range(num_classes)]
    recall = [report[str(i)]['recall'] for i in range(num_classes)]
    f1_score = [report[str(i)]['f1-score'] for i in range(num_classes)]
    # Get macro metrics from the classification report
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1_score = report['macro avg']['f1-score']

    # Calculate AUC-ROC and AUC-PR
    print("Shape of y_true (y_test_bin):", np.array(y_test_bin).shape)
    print("Shape of y_score (y_pred):", np.array(y_pred).shape)
    auc_roc = roc_auc_score(y_true=y_test_bin, y_score=y_pred,
                            multi_class='ovr',
                            average='weighted',
                            sample_weight=weights_test)
    # Calculate AUC-ROC OvO
    auc_roc_ovo = roc_auc_score(y_true=y_test_bin, y_score=y_pred,
                                multi_class='ovo',
                                average='weighted',
                                sample_weight=weights_test)

    auc_pr = average_precision_score(y_test_bin, y_pred,
                                     average='weighted',
                                     sample_weight=weights_test)

    # Calculate balanced accuracy
    balanced_accuracy = balanced_accuracy_score(y_true=y_test_le, y_pred=y_pred_labels,
                                                sample_weight=weights_test)

    # Calculate Cohen's Kappa
    cohen_kappa = cohen_kappa_score(y1=y_test_le, y2=y_pred_labels,
                                    sample_weight=weights_test)

    # Combine all metrics into a single DataFrame
    metrics = pd.DataFrame({
        'Macro Precision': macro_precision,
        'Macro Recall': macro_recall,
        'Macro F1 Score': macro_f1_score,
        'AUC-ROC OvR': [auc_roc] * num_classes,
        # 'AUC-ROC OvO': [auc_roc_ovo]*num_classes,
        'AUC-PR': [auc_pr] * num_classes,
        'Balanced Accuracy': [balanced_accuracy] * num_classes,
        'Cohen\'s Kappa': [cohen_kappa] * num_classes
    })
    return metrics


def plot_metrics(metrics_here, title, ax, show_legend=False, show_plot=False, save_path=None, save_name=None):
    # Check that, if save_path is not None, save_name is also provided
    assert (save_path is None) or (
                save_path is not None and save_name is not None), "If save_path is not None, save_name must also be provided"

    # Plot metrics
    ax = metrics_here.plot(kind='bar', ax=ax, colormap='tab10')

    # Set title and labels with LaTeX formatting
    ax.set_title(f'Classification Metrics: {title}', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)

    # Set y limit
    ax.set_ylim((0, 1))

    # Set xticks rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Move legend outside of plot area
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=12)
    # Show or hide legend based on show_legend parameter
    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=12)
    else:
        ax.get_legend().remove()
    if show_plot:
        # Ensure the plot fits into the figure area
        plt.tight_layout()
        # Show plot
        plt.show()
    # Get the figure associated with the Axes
    fig = ax.get_figure()
    # Save to png, pdf and svg (if save_path is not None):
    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(os.path.join(save_path, f'{save_name}.png'), dpi=600, bbox_inches='tight')
        fig.savefig(os.path.join(save_path, f'{save_name}.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(save_path, f'{save_name}.svg'), bbox_inches='tight')

    return ax, fig


def optimize_fun(trial, dtrain, dvalid, y_valid, num_classes, num_threads=os.cpu_count(), progress_bar=None):
    # Define the parameter space
    param = {
        'verbosity': 0,
        'silent': 1,
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-8, 1, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1, log=True),
    }

    if param['booster'] == 'gbtree' or param['booster'] == 'dart':
        param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
        param['eta'] = trial.suggest_float('eta', 1e-8, 1, log=True)
        param['gamma'] = trial.suggest_float('gamma', 1e-8, 1, log=True)
        param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        param['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10)  # Add this line

    my_params = {"objective": "multi:softprob",
                 "tree_method": "exact",
                 "num_class": num_classes,
                 'nthread': num_threads}

    param = {**param, **my_params}

    # Train the XGBoost model
    model = xgb.train(param, dtrain)

    # Predictions
    preds = model.predict(dvalid)

    # Check if 'preds' contains NaN values and skip optuna trial if so
    if np.isnan(preds).any():
        print("preds contains NaN values.")
        # update progress bar
        if progress_bar is not None:
            progress_bar.update(1)
        raise optuna.TrialPruned()  # Skip this trial

    # Calculate model's performance
    logloss = log_loss(y_valid, preds)  # Use log loss as the evaluation metric

    # update progress bar
    if progress_bar is not None:
        progress_bar.update(1)

    return logloss


def classification_benchmark(X_data, y_data, classification_type, num_classes,
                             seed=2023, test_size=0.2, n_br=100,
                             num_threads=os.cpu_count(),
                             n_trials=10,
                             optimization=True,
                             tree_method='exact',
                             # n_maxtrials=20
                             ):
    '''
    Perform classification on the input data using XGBoost.
    :param X_data: Feature data
    :param y_data: Target data
    :param classification_type: Type of classification: 'unbalanced' or 'weighted'
    :param seed: Random seed for reproducibility
    :param test_size: Fraction of the data to be used for testing
    :param n_br: Number of boosting rounds
    :param num_classes: Number of classes to classify
    :param num_threads: Number of threads to use
    :param n_trials: Number of optuna trials to optimize the XGBoost model
    :param optimization: Whether to perform optimization using optuna or skip it
    :param tree_method: Tree method to use with XGBoost: 'exact', 'approx' 'hist'
    :return: Classification metrics for the input data, figure axes, dictionary of classes, encoded test labels, and predicted probabilities
    '''
    assert classification_type in ['unbalanced',
                                   'weighted'], "classification_type must be either 'unbalanced' or 'weighted'"
    assert tree_method in ['exact', 'approx', 'hist'], "tree_method must be either 'exact', 'approx', or 'hist'"
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, stratify=y_data, test_size=test_size,
                                                        random_state=seed)
    # Encode labels
    le_train = LabelEncoder();
    le_test = LabelEncoder()
    le_train.fit(y_train);
    le_test.fit(y_test)
    # classes1 = le_train.classes_; classes2 = le_test.classes_
    y_train_le = le_train.transform(y_train)
    y_test_le = le_test.transform(y_test)
    print(y_train_le.shape, y_test_le.shape)
    # Get dictionary of classes
    classes = np.unique(y_data)
    classes_dict = {key: value for value, key in enumerate(classes)}
    print('classes_dict:', classes_dict)
    print('classes_dict.values:', classes_dict.values())
    print('classes_dict.keys:', classes_dict.keys())

    # Weighted case:
    if classification_type == 'weighted':
        # Create classification matrices
        class_weights = compute_class_weight('balanced', classes=classes, y=y_data)
        print("Class weights: ", class_weights)
        # Create a dictionary mapping class labels to their corresponding weights
        class_weight_dict = dict(zip(classes, class_weights))
        print("Class weight dictionary: ", class_weight_dict)
        # Define the weights for each class
        weights_train = np.zeros(len(y_train_le))
        weights_test = np.zeros(len(y_test_le))
        # Assign weights
        for i in classes_dict.values():
            weights_train[y_train_le == i] = class_weights[i]
            weights_test[y_test_le == i] = class_weights[i]

        # Create the DMatrix and include the weights
        dtrain_clf = xgb.DMatrix(X_train, label=y_train_le, weight=weights_train)
        dtest_clf = xgb.DMatrix(X_test, label=y_test_le, weight=weights_test)

    else:
        # Original data
        dtrain_clf = xgb.DMatrix(X_train, y_train_le, enable_categorical=True)
        dtest_clf = xgb.DMatrix(X_test, y_test_le, enable_categorical=True)
        weights_test = None

    # Parameter optimization
    my_params = {'objective': 'multi:softprob',
                 'booster': 'gbtree',
                 'tree_method': tree_method,
                 'num_class': num_classes,
                 'nthread': num_threads}

    if optimization:
        progress_bar = tqdm(total=n_trials, desc='Optimizing with optuna')
        # Create the Optuna study and run the optimization
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction='minimize')

        # fun params: trial,dtrain,dvalid,y_valid,num_classes=6,num_threads=os.cpu_count(),progress_bar=None
        study.optimize(lambda trial: optimize_fun(
            trial, dtrain_clf, dtest_clf, y_test_le, num_classes, num_threads, progress_bar),
                       n_trials=n_trials,
                       # callbacks=[optuna.study.MaxTrialsCallback(n_maxtrials, states=(optuna.trial.TrialState.COMPLETE,))]
                       )
        progress_bar.close()

        # Get parameters for xgboost
        params = study.best_params
        print('study_params = ', params)
        all_params = {**my_params, **params}
    else:
        all_params = my_params
    print('all_params = ', all_params)

    # Train the model
    print('Training the model...')
    model = xgb.train(all_params, dtrain_clf, num_boost_round=n_br, verbose_eval=True)
    # Make predictions on the test set
    print('Making predictions...')
    y_pred = model.predict(dtest_clf)
    # print('y_pred.shape =',y_pred.shape)
    # print('y_test_le.shape =', y_test_le.shape)
    # Calculate classification metrics for unbalanced case
    print('Calculating classification metrics...')
    metrics = classification_metrics(y_test_le, y_pred, num_classes=num_classes, weights_test=weights_test)
    metrics.index = classes_dict.keys()

    # Plot metrics
    # axs, figs = plot_metrics(metrics_here=metrics, title=title_plot,ax=ax,save_path=None,save_name='metrics_unbalanced',show_legend=True,show_plot=False)
    return model, metrics, y_test_le, y_pred, (X_train, X_test, y_train, y_test), all_params

def cancer_classification(histological_type_i, rnaseq, clinical_here, save_dir,
                          features=None,
                          n_threads=os.cpu_count(),
                          n_trials_optuna=100,
                          n_br=100,
                          classification_type='weighted',
                          stage_classification='i_ii_iii_iv',
                          possible_stages=['Stage I', 'Stage II', 'Stage III', 'Stage IV'],
                          scaler=None,
                          per=20,
                          test_size=0.2, preprocess=True, seed_here=np.random.randint(0, 1e6)):
    '''
    Classify cancer histological type. This function follows these steps:
    1. Extract corresponding stage and RNASeq information of given cancer
    2. Preprocessing: removing lowly-expressed and lowly-variance genes.
    3. Stage classification, based on the rnaseq data of the patients with the histological type  and the stage data from the clinical dataset.
    Classification is performed through xgboost, after optimizing the hyperparameters with optuna.

    :param histological_type_i: cancer type to classify
    :param rnaseq: rnaseq data for patients and genes
    :param clinical_here: clinical data, including histological type and stage
    :param save_dir: path to save the classification results
    :param features: features (genes) to select from rnaseq for classification
    :param n_threads: number of threads to use for parallel processing
    :param n_trials_optuna: number of optuna trials
    :param n_br: number of boosting rounds for xgboost
    :param classification_type: must begin with 'weighted' or 'unbalanced'
    :param stage_classification: type of classification to perform
    :param possible_stages: use only these stages for classification; can also be 'early' and 'late'
    :param scaler: scaler to use for data preprocessing or pipeline of scalers
    :param per: percentage of zeros in rnaseq to remove genes during preprocessing
    :param test_size: test size for classification
    :return:
    '''
    assert classification_type.startswith('weighted') or classification_type.startswith('unbalanced')
    assert stage_classification in ['i_ii_iii_iv', 'i_ii_iii', 'early_late']
    assert possible_stages in [['Stage I', 'Stage II', 'Stage III', 'Stage IV'], ['Stage I', 'Stage II', 'Stage III'],
                               ['early', 'late']]
    assert scaler is None or isinstance(scaler, StandardScaler) or isinstance(scaler, Pipeline) or isinstance(scaler,
                                                                                                              MinMaxScaler)
    assert isinstance(per, int) and per > 0 and per < 100
    # Set random seed
    # seed_here = np.random.randint(0,1e6)
    # Save path
    print(histological_type_i)
    save_dir_i = os.path.join(save_dir, histological_type_i)
    os.makedirs(save_dir_i, exist_ok=True)
    # Get clinical data of histological type i patients
    clinical_i = clinical_here[clinical_here['histological_type'] == histological_type_i]
    # Get rnaseq data of histological type i patients
    rnaseq_i = rnaseq.loc[:, rnaseq.columns.isin(clinical_i.index)]
    # Feature selection or data preprocessing
    if features is not None:
        rnaseq_i = rnaseq_i.loc[features]
        # Save rnaseq dataset:
        rnaseq_i.to_csv(os.path.join(save_dir_i, 'rnaseq_features.csv'))
    # Remove lowly-expressed genes:
    elif preprocess:
        # Keep genes with 0 expression in at least 20% of the patients
        rnaseq_redux = rnaseq_i[(rnaseq_i == 0).sum(axis=1) / rnaseq_i.shape[1] <= per / 100]
        # Keep genes whose mean expression and variance is equal or above 0.5 (in both cases).
        rnaseq_i = rnaseq_redux.iloc[(np.mean(rnaseq_redux, axis=1).values >= 0.5) &
                                     (np.var(rnaseq_redux, axis=1).values >= 0.5)]
        # Save rnaseq dataset:
        rnaseq_i.to_csv(os.path.join(save_dir_i, 'rnaseq_preprocessed.csv'))
    else:
        print('no data preprocessing')
    # Refine selected stages
    clinical_subset = clinical_i[clinical_i['ajcc_pathologic_tumor_stage'].isin(possible_stages)]
    num_classes = len(clinical_subset['ajcc_pathologic_tumor_stage'].unique())
    # Obtain corresponding rnaseq info
    rnaseq_subset = rnaseq_i.loc[:, np.isin(rnaseq_i.columns, clinical_subset.index)]
    # Check patients in clinical and rnaseq data coincides
    if clinical_subset.shape[0] != rnaseq_subset.shape[1]:
        clinical_subset = clinical_subset[clinical_subset.index.isin(rnaseq_subset.columns)]
    # Save clinical data
    clinical_subset.to_csv(os.path.join(save_dir_i, 'clinical_subset.csv'))
    # Check for no patients
    if clinical_subset.shape[0] == 0 or rnaseq_subset.shape[1] == 0:
        print(f'skipping {histological_type_i} classification because no patients were found')
        return
    # Scale rnaseq data
    if scaler is not None:
        cols_rnaseq = rnaseq_subset.columns
        rows_rnaseq = rnaseq_subset.index
        rnaseq_subset = scaler.fit_transform(rnaseq_subset)
        # Save scaled rnaseq dataset:
        pd.DataFrame(rnaseq_subset, columns=cols_rnaseq, index=rows_rnaseq).to_csv(
            os.path.join(save_dir_i, 'rnaseq_scaled.csv'))
    # Make classification
    classification_type_fun = 'weighted' if classification_type.startswith('weighted') else 'unbalanced'
    unbalanced_classification = classification_benchmark(
        X_data=rnaseq_subset.T,
        y_data=clinical_subset['ajcc_pathologic_tumor_stage'],
        classification_type=classification_type_fun,
        num_classes=num_classes,
        seed=seed_here,
        test_size=test_size,
        n_br=n_br,
        num_threads=n_threads,
        n_trials=n_trials_optuna,
    )

    return unbalanced_classification, save_dir_i, seed_here, classification_type_fun

