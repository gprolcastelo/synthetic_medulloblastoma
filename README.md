# README

## Overview
This project focuses on the analysis and visualization of medulloblastoma data using Variational Autoencoders (VAE). The pipeline includes data preprocessing, classification, clustering, and visualization steps.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Reproducing Results](#reproducing-results)
- [Code Organization](#code-organization)
- [Contributing](#contributing)
- [License](#license)

## Installation
To set up the project, you need to create a conda environment using the provided `environment.yml` file. 

*Note: using mamba instead of conda is recommended for faster environment creation.*

Follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/gprolcastelo/scratchmedulloblastoma.git
    cd scratchmedulloblastoma
    ```

2. Create the conda environment:
    ```bash
    conda env create -f environment.yml
    ```

3. Activate the environment:
    ```bash
    conda activate medulloblastoma
    ```

## Usage
After setting up the environment, you can start using the scripts provided in the repository. The main pipeline script is `pipeline.sh`, which runs the entire analysis pipeline.

After following the steps from the [Installation](#installation) sectionn, just execute `pipeline.sh`:

```bash
    bash pipeline.sh
```

## Reproducing Results

To reproduce the results, follow these steps:

1. Ensure that the conda environment is activated:
    ```bash
    conda activate medulloblastoma
    ```

2. Run the pipeline script:
    ```bash
    bash pipeline.sh
    ```

This script will execute all the necessary steps, including data preprocessing, classification, clustering, and visualization, and save the results to the `data` and `reports` subdirectories.

> **High Performance Computing (HPC) is highly recommended to run the pipeline**, as it requires significant computational resources, especially when training the VAE model and running the SHapley Additive exPlanations (SHAP) algorithm.


## Code Organization

The scripts are located in the `src` folder. The `pipeline.sh` bash scripts follows the analysis steps detailed in the paper. 

A brief summary of the scripts is provided:

0. `get_data.R` and `prepare_data.py` are used to download and prepare the data for usage, respectively. The preparation includes obtaining the gene names in different formats. 
The original data is not modified in this step.
Both of these scripts require an active internet connection.
1. `preprocessing.py` preprocesses the data.
2. `src/python_VAE.py` trains the VAE model for different hidden layers dimensions. GPU is recommended for this step.
3. `src/models/check_model.py` determines the VAE architecture with the lowest (best) reconstruction error.
4. `src/adjust_reconstruction.py` trains the postprocessing network on the output of the (best) VAE, determined in step 3.
5. `src/get_vae_outputs.py` saves to csv's data from the VAE's latent space, decoded data, and postprocessed data.
6. `src/group_classification.py` performs classification on the medulloblastoma subgroups in a given dataset.
7. `src/visualization/visualize.py` creates the UMAPs of the given dataset, plotting the different medulloblastoma subgroups.
8. `src/g3g4_clustering.R` performs consensus clustering to determine the patients in the G3-G4 subgroup.
9. `src/clustering_g3g4.py` performs clustering on the G3-G4 subgroup, following a kNNG with agglomerative clustering and bootstrapping approach.
10. `src/data_augmentation.py` uses the VAE's generative ability to augment the data in the specified subgroup(s).
11. `src/kruskalwallis_inbetween.py` performs the Kruskal-Wallis test on the synthetic data to determine singificantly differentially expressed genes.
12. `src/classification_shap.py` performs medulloblastoma subgroup classification in the latent space and uses SHAP to explain the model's predictions. 
The code maps the subgroup classification to latent space variables and these subsequently to the genes.
This is the most computationally expensive step to perform the explanations.
For us, this took about 18 hours on 112 cores.
13. `src/check_noise_ratio.py` checks the noise ratio in the synthetic data, comparing the genration of the synthetic data in the real data and the latent space plus reconstruction.
14. `src/genes_reconstruction.py` checks the reconstruction error of the genes and patients with Wasserstein distance, before and after applying the postprocessing network to the VAE-decoded data.
15. `python src/diff_genes_comparison.py` finds the overlap between differentially expressed genes and external data (Northcott et al. 2019 and Núñez-Carpintero et al. 2021).
16. `src/gprofiler.R` performs gene set enrichment analysis on the genes determined to be important in step 12.

Finally, the Jupyter Notebook `final_analyses.ipynb` contains the code for ad-hoc processes that were not included in the `src` folder codes.


## Contributing
Contributions generally not expected. However, if you have any suggestions or improvements, feel free to reach out in the issues section. 
On the other hand, this is an open-source, open-science project, and so you are encouraged to reproduce our results.
Feel free to clone or fork the repository.

## License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.
