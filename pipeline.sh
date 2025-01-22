#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
today=$(date +"%Y%m%d")
current_dir=$(pwd)

# Paths to data and metadata:
data_path="${current_dir}/data/raw/GEOquery/GSE85217_expression_data.csv"
metadata_path="${current_dir}/data/raw/GEO/cavalli_subgroups.csv"

# Define the model name and paths to save the results
model="VAE"
batch_size=8
model_path="${current_dir}/models/${today}_${model}"
path_to_vae_results="${current_dir}/data/interim/${today}_${model}_batches"
output_rec_path="${current_dir}/data/interim/${today}_${model}_adjust_reconstruction"
preprocessing_path="${current_dir}/data/interim/${today}_preprocessing"
preprocessed_data_path="${current_dir}/data/interim/${today}_preprocessing/cavalli_maha.csv"
mkdir -p err_out
mkdir -p $model_path

# 0. Obain data
Rscript get_data.R
python prepare_data.py
exit 1
# 1. Preprocessing
python src/preprocessing.py --data_path $data_path \
							--metadata_path $metadata_path \
							--save_path $preprocessing_path \
							--per 0.2 \
							--cutoff 0.1 \
							--alpha 0.05

# 2. Train VAE with a combination of hyperparameters
for md in 256 512 1024 2048 4096; do
  for f in 8 16 32 64 128 256 512; do
    for lr in 0.00001 0.0001 0.001; do
      python src/python_VAE.py --md $md \
                                                   --f $f \
                                                   --lr $lr \
                                                   --path_rnaseq $preprocessed_data_path \
                                                   --path_clinical $metadata_path \
                                                   --save_path $path_to_vae_results \
                                                   --save_model \
                                                   --save_model_path $model_path \
                                                   --batch_size $batch_size
    done
  done
done

# 3. Get the best model
python src/models/check_model.py --save_path $path_to_vae_results \
                                 --rnaseq_path $preprocessed_data_path \
                                 --clinical_path $metadata_path
## Path to the CSV file
csv_file="${path_to_vae_results}/best_params.csv"

## Read the CSV file and extract the values of 'md' and 'f' from the first row
idim=$(awk -F, 'NR==2 {print $2}' "$csv_file")
md=$(awk -F, 'NR==2 {print $3}' "$csv_file")
f=$(awk -F, 'NR==2 {print $4}' "$csv_file")
lr=$(awk -F, 'NR==2 {print $5}' "$csv_file")
## Construct the string and save it to a variable
path_to_best_model="${current_dir}/models/${today}_${model}/${today}_${model}_idim${idim}_md${md}_feat${f}_lr${lr}.pth"

## Print the result
echo "The best model is located at:"
echo "${path_to_best_model}"


# 4. Train reconstruction network with the best model
adjust_path=${current_dir}/data/interim/${today}_adjust_reconstruction
recnet_path=${current_dir}/models/${today}_adjust_reconstruction/network_reconstruction.pth
python src/adjust_reconstruction.py --model_path $path_to_best_model \
                                    --data_path $preprocessed_data_path \
                                    --clinical_path $metadata_path \
                                    --output_path $adjust_path \
                                    --output_recnet_path $recnet_path \
                                    --batch_size 8 \
                                    --test_size 0.2 \
                                    --seed 2023 \
                                    --epochs 1000 \
                                    --lr 0.0001 \
                                    --n_trials 100

# 5. Generate latent space data, decoded data, and postprocessed data
vae_data_path="${current_dir}/data/processed/${today}_vae_output"
python src/get_vae_outputs.py --data_path $preprocessed_data_path \
                             --metadata_path $metadata_path \
                             --model_path $path_to_best_model \
                             --recnet_path $recnet_path \
                             --recnet_hyperparams_path ${adjust_path}/best_hyperparameters.csv \
                             --output_path $vae_data_path \
                             --seed 2023

# 6. Classification on real, latent, decoded, and reconstructed data
## Real data
python src/group_classification.py --path_data $data_path \
              --path_metadata $metadata_path \
              --save_dir ${current_dir}/reports/figures/${today}_classification_original \
              --n_classes 4 \
              --classification_type weighted \
              --n_seeds 10 \
              --n_trials_optuna 100 \
              --n_br 100 \
              --test_size 0.2 \
              --n_threads 112
## Preprocessed data
python src/group_classification.py --path_data $preprocessed_data_path \
              --path_metadata $metadata_path \
              --save_dir ${current_dir}/reports/figures/${today}_classification_preprocessed \
              --n_classes 4 \
              --classification_type weighted \
              --n_seeds 10 \
              --n_trials_optuna 100 \
              --n_br 100 \
              --test_size 0.2 \
              --n_threads 112
## Latent space
python src/group_classification.py --path_data ${vae_data_path}/z.csv \
              --path_metadata $metadata_path \
              --save_dir ${current_dir}/reports/figures/${today}_classification_latent \
              --n_classes 4 \
              --classification_type weighted \
              --n_seeds 10 \
              --n_trials_optuna 100 \
              --n_br 100 \
              --test_size 0.2 \
              --n_threads 112
## Postprocessed data
python src/group_classification.py --path_data ${vae_data_path}/postprocessed.csv \
              --path_metadata $metadata_path \
              --save_dir ${current_dir}/reports/figures/${today}_classification_postprocessed \
              --n_classes 4 \
              --classification_type weighted \
              --n_seeds 10 \
              --n_trials_optuna 100 \
              --n_br 100 \
              --test_size 0.2 \
              --n_threads 112


# 7. Umaps from all spaces
## Real data
python src/visualization/visualize.py --data_path $data_path \
                                      --metadata_path $metadata_path \
                                      --save_path ${current_dir}/reports/figures/${today}_umap_original \
                                      --groups "SHH, WNT, Group 3, Group 4" \
                                      --n_components 2 \
                                      --seed 2023
## Preprocessed data
python src/visualization/visualize.py --data_path $data_path \
                                      --metadata_path $metadata_path \
                                      --save_path ${current_dir}/reports/figures/${today}_umap_preprocessed \
                                      --groups "SHH, WNT, Group 3, Group 4" \
                                      --n_components 2 \
                                      --seed 2023
## Latent space
python src/visualization/visualize.py --data_path ${vae_data_path}/z.csv \
                                      --metadata_path $metadata_path \
                                      --save_path ${current_dir}/reports/figures/${today}_umap_latent \
                                      --groups "SHH, WNT, Group 3, Group 4" \
                                      --n_components 2 \
                                      --seed 2023
## Postprocessed data
python src/visualization/visualize.py --data_path ${vae_data_path}/postprocessed.csv \
                                      --metadata_path $metadata_path \
                                      --save_path ${current_dir}/reports/figures/${today}_umap_postprocessed \
                                      --groups "SHH, WNT, Group 3, Group 4" \
                                      --n_components 2 \
                                      --seed 2023

## 8. Consensus clustering
## Original data
Rscript src/g3g4_clustering.R --data_path "$data_path" \
                                --metadata_path "$metadata_path" \
                                --clusterAlg "km" \
                                --real_space \
                                --save_path "${current_dir}/data/processed/${today}_consensusclustering/results_real_space" \
                                --plot_ext "pdf" \
                                --reps 1000 \
                                --psamples 0.8 \
                                --pfeatures 1 \
                                --maxk 3

## Preprocessed data
Rscript src/g3g4_clustering.R --data_path "$data_path" \
                                --metadata_path "$metadata_path" \
                                --clusterAlg "km" \
                                --real_space \
                                --save_path "${current_dir}/data/processed/${today}_consensusclustering/results_preprocessed" \
                                --plot_ext "pdf" \
                                --reps 1000 \
                                --psamples 0.8 \
                                --pfeatures 1 \
                                --maxk 3

# 9. knn with bootstrapping
metadata_path_knn_latent=${current_dir}/data/processed/${today}_knn_bootstrap_latent/metadata_after_bootstrap.csv
## Latent space
python src/clustering_g3g4.py --data_path ${vae_data_path}/z.csv \
                              --metadata_path $metadata_path  \
                              --save_path ${current_dir}/data/processed/${today}_knn_bootstrap_latent

## Real space
python src/clustering_g3g4.py --data_path $preprocessed_data_path \
                              --metadata_path $metadata_path  \
                              --save_path ${current_dir}/data/processed/${today}_knn_bootstrap_preprocessed
## Umap of the real space with the knn-detected groups
### Original data
python src/visualization/visualize.py --data_path $data_path \
                                      --metadata_path $metadata_path_knn_latent \
                                      --save_path ${current_dir}/data/processed/${today}_knn_bootstrap_latent/knn_bootstrapping/original_umap \
                                      --groups "SHH WNT Group3 Group4 G3-G4" \
                                      --n_components 2 \
                                      --seed 2023
### Preprocessed data
python src/visualization/visualize.py --data_path $preprocessed_data_path \
                                      --metadata_path $metadata_path_knn_latent \
                                      --save_path ${current_dir}/data/processed/${today}_knn_bootstrap_latent/knn_bootstrapping/preprocessed_umap \
                                      --groups "SHH WNT Group3 Group4 G3-G4" \
                                      --n_components 2 \
                                      --seed 2023


# 10. Data augmentation to balance the groups: with knn from real and latent spaces
trap 'cleanup' SIGTERM
n_synth=200
metadata_path_after_bootstrap_real=${current_dir}/data/processed/${today}_knn_bootstrap_preprocessed/metadata_after_bootstrap.csv
metadata_path_after_bootstrap_latent=${current_dir}/data/processed/${today}_knn_bootstrap_latent/metadata_after_bootstrap.csv
save_path_augment=${current_dir}/data/interim/${today}_data_augmentation
# For groups: Group 3, Group 4, and G3-G4
python src/data_augmentation.py --data_path $preprocessed_data_path \
                           --clinical_path $metadata_path_after_bootstrap_real \
                           --model_path $path_to_best_model \
                           --network_model_path $recnet_path \
                           --recnet_hyperparams_path ${adjust_path}/best_hyperparameters.csv \
                           --mu 0 \
                           --std 1 \
                           --noise_ratio 0.25 \
                           --group_to_augment "Group 3, Group 4, G3-G4" \
                           --n_synth $n_synth \
                           --results_path ${save_path_augment}/real


# 11. Kruskal-Wallis test on synth data
alpha_kw=0.01
kw_boxplot_path=${current_dir}/reports/figures/${today}_kw/boxplot_augmented
kw_path=${current_dir}/data/processed/${today}_differentially_expressed_genes

python src/kruskalwallis_inbetween.py --path_data ${save_path_augment}/real/augmented_data.csv \
                                                                --path_clinical ${save_path_augment}/real/augmented_clinical.csv \
                                                                --path_genes "all" \
                                                                --alpha $alpha_kw \
                                                                --path_boxplot $kw_boxplot_path/synth_patients \
                                                                --save_path $kw_path/synth_patients \
                                                                --group_to_analyze "synthetic_Group 3, synthetic_Group 4, synthetic_G3-G4"

# 12. SHAP
shap_save_path=${current_dir}/data/interim/${today}_shap

python src/classification_shap.py --n_shap 100 \
                                  --qval 0.95 \
                                  --num_classes 4 \
                                  --n_trials 100 \
                                  --n_br 100 \
                                  --tree_method exact \
                                  --data_path $preprocessed_data_path \
                                  --clinical_path $metadata_path \
                                  --model_path $path_to_best_model \
                                  --save_path ${shap_save_path}/real_data \
                                  --group_to_analyze "all"
echo "Done with the pipeline."

# 13. Check noise ratio
python src/check_noise_ratio.py --data_path $preprocessed_data_path \
                                --clinical_path $metadata_path \
                                --model_path $path_to_best_model \
                                --recnet_path $recnet_path \
                                --hyperparam_path ${adjust_path}/best_hyperparameters.csv \
                                --save_path ${current_dir}/reports/figures/${today}_noise_ratio

# 14. Check the reconstruction error with Wasserstein distance
python src/genes_reconstruction.py --data_path $preprocessed_data_path \
                                --clinical_path $metadata_path \
                                --model_path $path_to_best_model \
                                --recnet_path $recnet_path \
                                --hyperparam_path ${adjust_path}/best_hyperparameters.csv \
                                --save_path ${current_dir}/reports/figures/${today}_reconstruction_error

# 15. Overlap of genes with external data
python src/diff_genes_comparison.py --external_genes data/external/Supplementary_Table_5.csv \
                                                               --internal_genes_differential ${kw_path}/synth_patients/always_diff_genes.csv \
                                                               --group_of_interest G3_G4 \
                                                               --save_path data/processed/${today}_genes_comparison/real_patients

# 16. Enrichment analysis with gprofiler
Rscript src/gprofiler.R ${shap_save_path}/selected_genes.csv ${current_dir}/data/processed/${today}_gprofiler_enrichment

echo "Done with the pipeline."
