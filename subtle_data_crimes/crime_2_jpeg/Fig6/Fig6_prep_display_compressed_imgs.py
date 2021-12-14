'''
This code presents the example shown in Fig 6 in the paper.

Notice: the code loads the MoDL networks that were trained for the experiments shown in Fig 7.
So before running this code, you should:
1. Go to the folder Fig7/DL/ and train the networks (or download our pre-trained networks).
2. In that folder, run the code Test_MoDL_R4_forFig6.py - it will save results that are loaded here.

NOTICE: in this code, you should update the following variables to YOUR desired path (see first code cell):
FastMRI_train_folder    # input folder
FastMRI_val_folder      # input folder
FatSat_processed_data_folder  # desired output folder

In this script and all the associated scripts:
q is the JPEG quality factor.
q=999 is used for saving data WITHOUT JPEG compression.

(c) Efrat Shimron (UC Berkeley, 2021)
'''


import numpy as np
import matplotlib.pyplot as plt

print('loading results... be patient...')
container = np.load('../Fig7_jpeg_statistics/DL/Res_for_Fig6.npz')
TARGETS = container['TARGETS']
RECS = container['RECS']
R_vec = container['R_vec']
q_vec = container['q_vec']
checkpoint_num = container['checkpoint_num']
NRMSE_av_vs_q_and_R = container['NRMSE_av_vs_q_and_R']
NRMSE_std_vs_q_and_R = container['NRMSE_std_vs_q_and_R']
SSIM_av_vs_q_and_R = container['SSIM_av_vs_q_and_R']
SSIM_std_vs_q_and_R = container['SSIM_std_vs_q_and_R']
NRMSE_examples_4display = container['NRMSE_examples_4display']
SSIM_examples_4display = container['SSIM_examples_4display']


# Dimensions of loaded arrays
# TARGETS = [NX, NY, q_vec.shape[0], N_examples_4display]
# RECS = [NX, NY, R_vec.shape[0], q_vec.shape[0], N_examples_4display]
# NRMSE_examples_4display = [R_vec.shape[0],q_vec.shape[0],N_calc_err]
# SSIM_examples_4display = [R_vec.shape[0],q_vec.shape[0],N_calc_err]


N_examples_4display = 1

NX = TARGETS.shape[0]
NY = TARGETS.shape[1]

R_vec = np.array([4])

q_vec_4display = np.array([20,50,75,999])
Nq = q_vec_4display.shape[0]

for n in range(N_examples_4display):

    for r in range(R_vec.shape[0]):
        R = R_vec[r]

        n_example = 14

        # display full-FOV image
        im_target0 = TARGETS[:, :, -1, n_example]
        fig = plt.figure()
        plt.imshow(np.abs(im_target0),cmap="gray")
        plt.axis('off')
        plt.clim(0,2.2)
        plt.colorbar()
        plt.show()
        figname = 'gold_example_{}'.format(n_example)
        fig.savefig(figname)


        fig = plt.figure(figsize=(13, 5))
        for q_i in range(q_vec_4display.shape[0]):
            q = q_vec_4display[q_i]
            print('q=',q)
            q_i_in_loaded_array = np.where(q_vec==q)[0][0]
            #print('q={} q_i_loaded_array={}'.format(q,q_i_in_loaded_array))
            im_target = TARGETS[:,:,q_i_in_loaded_array,n_example].squeeze()
            im_rec = RECS[:,:,r,q_i_in_loaded_array,n_example].squeeze()
            im_tar_zoomed = im_target[270:400, 70:290]
            im_rec_zoomed = im_rec[270:400, 70:290]
            NX_zoomed = im_tar_zoomed.shape[1]
            NY_zoomed = im_tar_zoomed.shape[0]

            plt.subplot(2,Nq,(Nq-q_i))
            plt.imshow(im_tar_zoomed,cmap="gray")
            plt.axis('off')
            plt.clim(0, 2.2)
            plt.title('target q={}'.format(q))

            plt.subplot(2,Nq,2*Nq-q_i)
            plt.imshow(im_rec_zoomed,cmap="gray")
            plt.axis('off')
            plt.clim(0, 2.2)
            CS_str = 'NRMSE {:.4f}'.format(NRMSE_examples_4display[r,q_i_in_loaded_array,n])
            plt.text(0.02 * NX_zoomed, 0.95 * NY_zoomed, CS_str, color="yellow",fontsize=15)

        plt.suptitle('Example #{} - R={}'.format(n_example,R))
        plt.show()
        figname = 'rec_fig_R{}_example_{}'.format(R,n_example)
        fig.savefig(figname)
