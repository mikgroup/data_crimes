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

# container_2 = np.load('RECS_and_TARGETS.npz')
# TARGETS = container_2['TARGETS']
# RECS = container_2['RECS']

# structure of arrays - copied from Test_MoDL_on_jpeg_data.py
#TARGETS = np.zeros((NX, NY, q_vec.shape[0], N_examples_4display))
#RECS = np.zeros((NX, NY, R_vec.shape[0], q_vec.shape[0], N_examples_4display))
#NRMSE_examples_4display = np.zeros((R_vec.shape[0],q_vec.shape[0],N_calc_err))
#SSIM_examples_4display = np.zeros((R_vec.shape[0],q_vec.shape[0],N_calc_err))

print('preparing figs...')
N_examples_4display = 1 #50 #TARGETS.shape[3]

NX = TARGETS.shape[0]
NY = TARGETS.shape[1]

R_vec = np.array([4])

q_vec_4display = np.array([20,50,75,999])
Nq = q_vec_4display.shape[0]

for n in range(N_examples_4display):

    for r in range(R_vec.shape[0]):
        R = R_vec[r]

        n_example = 14

        # # display full-FOV image (q=100)
        im_target0 = TARGETS[:, :, -1, n_example]
        fig = plt.figure()
        plt.imshow(np.abs(im_target0),cmap="gray")
        plt.axis('off')
        plt.clim(0,2.2)
        plt.colorbar()
        plt.show()
        figname = 'gold_example_{}'.format(n_example)
        fig.savefig(figname)

        #fig = plt.figure(figsize=(17,5)) # for q_vec=[10,20,50,75,100]
        fig = plt.figure(figsize=(13, 5))
        #for q_i in range(Nq):
        #    q = q_vec[q_i]
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
            #plt.imshow(im_target, cmap="gray")
            plt.axis('off')
            plt.clim(0, 2.2)
            plt.title('target q={}'.format(q))

            plt.subplot(2,Nq,2*Nq-q_i)
            plt.imshow(im_rec_zoomed,cmap="gray")
            #plt.title('target q={}'.format(q))
            #plt.imshow(im_rec, cmap="gray")
            plt.axis('off')
            plt.clim(0, 2.2)
            CS_str = 'NRMSE {:.4f}'.format(NRMSE_examples_4display[r,q_i_in_loaded_array,n])
            plt.text(0.02 * NX_zoomed, 0.95 * NY_zoomed, CS_str, color="yellow",fontsize=15)
            #plt.title('NRMSE {:.4f}'.format(NRMSE_examples_4display[r,q_i_in_loaded_array,n]))


        plt.suptitle('Example #{} - R={}'.format(n_example,R))
        plt.show()
        figname = 'rec_fig_R{}_example_{}'.format(R,n_example)
        fig.savefig(figname)

print('')
