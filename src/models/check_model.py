import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def fun_filename_from_params_1hdl(mdim,fea,lear,my_path):
    string = "md"+str(mdim)+"_f"+str(fea)+"_lr"+str(lear)
    list_of_paths = [i for i in os.listdir(my_path)]
    string_in_list =[string in i for i in list_of_paths]
    if sum(string_in_list)==0:
        return []

    pathname = np.array(list_of_paths,dtype=str)[string_in_list][0]
    return os.path.join(my_path, pathname)# + ".py_loss.csv"


def extract_param_from_pathname_1hl(my_path):
    import re
    print('my_path:',my_path)
    mid_dim, feat, lr = [], [], []
    # loop through all directories (excluding files):
    for i in next(os.walk(my_path))[1]:
        params_i = [str(s) for s in re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", i)[1:]]
        mid_dim.append(int(params_i[0]))
        feat.append(int(params_i[1]))

        lr.append(str(params_i[2]))

    mid_dim = np.unique(mid_dim)
    feat = np.unique(feat)
    lr = np.unique(lr)

    lr = np.sort(lr)[::-1]

    return mid_dim, feat, lr


def collect_data_VAE_1hl(path_collect, filename, epochs_mean, md, f, lr):
    Mat = []
    mat_train = []
    mat_test = []
    for i in md:
        mat_i = []
        mat_traini = []
        mat_testi = []
        for j in f:
            mat_j = []
            mat_trainj = []
            mat_testj = []
            for k in lr:
                mat_l = [i, j, k]
                mat_j.append(mat_l)
                path_to_loss = fun_filename_from_params_1hdl(i, j, k, path_collect)

                path_to_loss = os.path.join(path_to_loss, filename)

                if path_to_loss == []:
                    example = np.nan
                    mat_trainj.append(example)
                    mat_testj.append(example)
                else:

                    example = pd.read_csv(path_to_loss, index_col=0)
                    mat_trainj.append(np.mean(example.iloc[:, [0]].values[-epochs_mean:]))
                    mat_testj.append(np.mean(example.iloc[:, [1]].values[-epochs_mean:]))

            mat_i.append(mat_j)
            mat_traini.append(mat_trainj)
            mat_testi.append(mat_testj)

        Mat.append(mat_i)
        mat_train.append(mat_traini)
        mat_test.append(mat_testi)

    Mat = np.array(Mat)
    mat_train = np.array(mat_train)
    mat_test = np.array(mat_test)

    return Mat, mat_train, mat_test



def check_trained_vae(path_here, save_path,epochs_mean=20,plot_heatmap=True):
    list_of_paths = os.listdir(path_here)
    md, f, lr = extract_param_from_pathname_1hl(path_here)
    Mat, mat_train, mat_test = collect_data_VAE_1hl(
        path_collect=path_here,
        filename=f"rec_loss.csv",
        epochs_mean=epochs_mean, md=md, f=f, lr=lr)
    # Get a heatmap of the reconstruction loss for each learning rate:
    if plot_heatmap:
        for i in range(len(lr)):
            data_i = np.array(mat_test)[:, :, i]
            df_i = pd.DataFrame(
                data_i,
                columns=f,
                index=md#[::-1]
            )
            # Save csv:
            df_i.to_csv(os.path.join(save_path, f"losstable_lr{lr[i]}.csv"))
            # Save latex table
            df_i.to_latex(os.path.join(save_path, f"losstable_lr{lr[i]}.tex"))
            # Start figure
            plt.figure(figsize=(9, 9))
            plt.imshow(np.array(mat_test)[:, :, i],
                       cmap="plasma",
                       origin='lower',
                       extent=[0, max(f), 0, max(md)],
                       aspect="auto",
                       interpolation='none'
                       )
            plt.grid(False)  # Disable the grid
            plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            plt.title("Reconstruction Loss with lr = " + str(lr[i]) + "\n", fontsize=18)
            plt.xlabel("Latent Space Dimension", fontsize=16)
            plt.ylabel("Hidden Layer Dimension", fontsize=16)
            plt.yticks([max(md) / len(md) / 2 + i * max(md) / len(md) for i in range(len(md))],
                       labels=md, fontsize=14)
            plt.xticks([max(f) / len(f) / 2 + i * max(f) / len(f) for i in range(len(f))],
                       labels=f, fontsize=14)

            cax = plt.axes([0.9, 0.15, 0.1, 0.7])
            plt.colorbar(cax=cax).set_label(label='\nTest Loss (nats)', size=16)

            # Save the figure:
            plt.savefig(os.path.join(save_path, "heatmap_lr" + str(lr[i]) + ".png"), bbox_inches='tight', dpi=600)
            plt.savefig(os.path.join(save_path, "heatmap_lr" + str(lr[i]) + ".svg"), bbox_inches='tight')
            plt.savefig(os.path.join(save_path, "heatmap_lr" + str(lr[i]) + ".pdf"), bbox_inches='tight')

    # Get minimum test loss parameters:
    print("Minimum test loss found:",np.nanmin(mat_test))
    pos_min = np.where(mat_test == np.nanmin(mat_test))
    min_md = md[pos_min[0][0]]
    min_f = f[pos_min[1][0]]
    min_lr = lr[pos_min[2][0]]

    return min_md, min_f, min_lr

def main(args):
    min_md, min_f, min_lr = check_trained_vae(
        path_here=args.save_path,
        save_path=args.save_path,
        epochs_mean=args.epochs_mean)
    print(f"Minimum test loss found for: md={min_md}, f={min_f}, lr={min_lr}")
    # Load the RNASeq data to get the input dimension:
    rnaseq = pd.read_csv(args.rnaseq_path, index_col=0)
    clinical = pd.read_csv(args.clinical_path, index_col=0)
    # Make sure rnaseq and clinical have the same first index (which is the number of samples):
    if rnaseq.shape[0] != clinical.shape[0]:
        if rnaseq.shape[1] == clinical.shape[0]:
            rnaseq = rnaseq.T
        elif rnaseq.shape[0] == clinical.shape[1]:
            clinical = clinical.T
        else:
            raise ValueError("RNA-seq and clinical data do not have the same number of samples.")
    # Idim then is the number of features (genes):
    idim = rnaseq.shape[1]
    # Save the best parameters as csv:
    df_best = pd.DataFrame(data={"idim":[idim],"md": [min_md], "f": [min_f], "lr": [min_lr]})
    df_best.to_csv(os.path.join(args.save_path, "best_params.csv"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check the best model parameters for a VAE')
    parser.add_argument('--save_path', type=str, help='Path to the directory containing the trained models')
    parser.add_argument('--epochs_mean', type=int, default=20, help='Number of final epochs to average the loss over')
    parser.add_argument('--rnaseq_path', type=str, help='Path to the RNASeq data. Only used to save idim.')
    parser.add_argument('--clinical_path', type=str, default=True, help='Path to metadata. Only used to check number of samples.')
    args = parser.parse_args()
    main(args)