import numpy as np
import matplotlib.pyplot as plt
import os

################################### settings ####################################

R_vec = np.array([2,3,4])
N_examples = 122
var_dens_flag = 'weak'
N_methods = 3 # 0 = CS, 1 = DictL, 2 = DNN

figs_path = 'stats_figs'

if not os.path.exists(figs_path):
    os.makedirs(figs_path)

######################################## load CS & DictL results ###########################################
print('loading CS & DictL results...')
res_filename = f'CS_and_DictL/RESULTS_CS_DictL_{var_dens_flag}_VD_N_examples_{N_examples}.npz'

container = np.load(res_filename)


CS_NRMSE_array=container['CS_NRMSE_array']
CS_SSIM_array=container['CS_SSIM_array']

DictL_NRMSE_array=container['DictL_NRMSE_array']
DictL_SSIM_array=container['DictL_SSIM_array']
q_vec_from_CS_DictL = container['q_vec']

# reminder - the arrays structure (from the code 2_load_and_aggregate_results_CS_DictL.py):
# CS_NRMSE_array = np.empty([q_vec.shape[0], R_vec.shape[0], N_examples])
# CS_SSIM_array = np.empty([q_vec.shape[0], R_vec.shape[0], N_examples])
# DictL_NRMSE_array = np.empty([q_vec.shape[0], R_vec.shape[0], N_examples])
# DictL_SSIM_array = np.empty([q_vec.shape[0], R_vec.shape[0], N_examples])

# ############################## Load DL results ###################################
print('loading DNN results...')
DL_container = np.load('DL/Res_MODL_JPEG_exp.npz')

DL_NRMSE_array = DL_container['DL_NRMSE_array']
DL_SSIM_array = DL_container['DL_SSIM_array']
q_vec_from_DL_container = DL_container['q_vec']


################ reorganize the data - keep only results corresponding to q=25,50,75,999 ####################3
# The next code deals with cases in which the CS, DictL and DL runs were performed with additional q values (e.g. q=90).
# Here we keep only the results that we wish to display in graphs later. For that, we first find the relevant indices.
# note: q=999 corresponds to the case of No Compression (NC), and it will be mapped to q=101 just for display purposes

q_vec = np.array([20,50,75,999])
q_inds_CS_DictL = []
q_inds_DL = []

for q_tmp in q_vec:
    # find indecies for CS and DictL runs
    qi_CS = np.argwhere(q_vec==q_tmp)
    q_inds_CS_DictL = np.append(q_inds_CS_DictL,qi_CS)
    # find indices for DL runs
    qi_DL = np.argwhere(q_vec_from_DL_container==q_tmp)
    q_inds_DL= np.append(q_inds_DL,qi_DL)


q_inds_CS_DictL = q_inds_CS_DictL.astype(int) # convert to int
q_inds_DL = q_inds_DL.astype(int)

# remove unnecessary values from the NRMSE & SSIM arrays
CS_NRMSE_array = CS_NRMSE_array[q_inds_CS_DictL,:,:]
CS_SSIM_array = CS_SSIM_array[q_inds_CS_DictL,:,:]
DictL_NRMSE_array = DictL_NRMSE_array[q_inds_CS_DictL,:,:]
DictL_SSIM_array = DictL_SSIM_array[q_inds_CS_DictL,:,:]

DL_NRMSE_array = DL_NRMSE_array[q_inds_DL,:,:]
DL_SSIM_array = DL_SSIM_array[q_inds_DL,:,:]

############################################# display ##########################################################

print('preparing figures...')

# ----------- preparation for plots -----------------
x_ticks_labels = []
for i in range(q_vec.shape[0]):

    if q_vec[(-i-1)]==999:
        q_label='NC'
    else:
        q_label = q_vec[(-i - 1)]
    x_ticks_labels.append(f'{q_label}')

print(x_ticks_labels)

markers=['o', 'd', 's', '^','x']
colorslist = ['darkcyan','royalblue','darkcyan','k']

#q_vec_for_plot = q_vec.copy()
q_vec_for_plot = q_vec
q_vec_for_plot[q_vec==999] = 101

print('q_vec=',q_vec)
print('q_vec_for_plot=',q_vec_for_plot)


# ------------------ figures ------------------------

for method_i in range(N_methods):
    if method_i==0:
        method_str = 'CS'
        NRMSE_arr = CS_NRMSE_array
        SSIM_arr = CS_SSIM_array
    elif method_i==1:
        method_str = 'DictL'
        NRMSE_arr = DictL_NRMSE_array
        SSIM_arr = DictL_SSIM_array
    elif method_i == 2:
         method_str = 'DL'
         NRMSE_arr = DL_NRMSE_array # TODO: update this
         SSIM_arr = DL_SSIM_array # TODO: update this

    print(f'--------- {method_str} --------')


    ############ NRMSE figure ###############


    # NRMSE figure
    fig = plt.figure()
    for r in range(R_vec.shape[0]):
        R = R_vec[r]
        label_str = f'R={R}'


        NRMSE_vec = NRMSE_arr[:, r, :].squeeze()
        NRMSE_av_per_q_N = NRMSE_vec.mean(axis=1)
        NRMSE_std_per_q_N = NRMSE_vec.std(axis=1)

        #print(f'R={R} NRMSE ',NRMSE_av_per_q_N)

        # compute the change between the cases of No-Compression (NC, q=101) to high compression (q=20)
        qNC_i = np.argwhere(q_vec == 101)
        q20_i = np.argwhere(q_vec == 20)

        NRMSE_diff = NRMSE_av_per_q_N[q20_i] - NRMSE_av_per_q_N[qNC_i]
        NRMSE_drop = NRMSE_diff / NRMSE_av_per_q_N[qNC_i] * 100
        #print(f'{method_str}  R{R} NRMSE decrease from NC to 20: {NRMSE_drop}%')

        # print results for table 2 in the paper:
        print(f'R={R} NRMSE init {NRMSE_av_per_q_N[-1]:.4f}; NRMSE end {NRMSE_av_per_q_N[0]:.4f} ; drop {NRMSE_drop}%')

        if R==np.max(R_vec):
            # plot lines between NC and q20
            # plot horizontal line
            plt.plot(np.array([100, 20]), (NRMSE_av_per_q_N[qNC_i].item(), NRMSE_av_per_q_N[qNC_i].item()), 'k--')

            # plot vertical line
            plt.plot(np.array([20, 20]), np.array([NRMSE_av_per_q_N[q20_i].item(), NRMSE_av_per_q_N[qNC_i].item()]),
                     'k--')



        plt.plot(q_vec, NRMSE_av_per_q_N, linestyle='solid', label=label_str,color=colorslist[r],
                     marker=markers[r])

        plt.fill_between(q_vec, (NRMSE_av_per_q_N - NRMSE_std_per_q_N/2), (NRMSE_av_per_q_N + NRMSE_std_per_q_N/2), color=colorslist[r],alpha=0.1)

    plt.xlabel('JPEG quality', fontsize=20)
    plt.ylabel('NRMSE', fontsize=20)
    ax = plt.gca()
    ax.set_xticks(q_vec_for_plot[::-1])
    ax.set_xticklabels(x_ticks_labels, fontsize=14)
    ax.invert_xaxis()
    if np.max(R_vec)==3:
        plt.ylim(0.002, 0.017)
        ax.set_yticks((0.005, 0.01, 0.015))
    elif np.max(R_vec)==4:
        plt.ylim(0.002, 0.022)
        ax.set_yticks((0.005, 0.01, 0.015, 0.02))
    elif np.max(R_vec)==6:
        plt.ylim(0.002, 0.045)
        #ax.set_yticks((0.005,0.01, 0.015,0.02))


    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.title(method_str)
    if (method_i==(N_methods-1)):
        ax.legend(fontsize=15,loc='upper right')
    #ax.grid('on')d
    plt.title(f'{method_str} - {var_dens_flag} VD {N_examples} examples')
    plt.show()
    figname_NRMSE = figs_path + f'/{method_str}_NRMSE_stats'
    fig.savefig(fname=figname_NRMSE)


    ############################################################################
    # SSIM figure
    fig = plt.figure()
    for r in range(R_vec.shape[0]):
        R = R_vec[r]
        label_str = f'R={R}'

        SSIM_vec = SSIM_arr[:, r, :].squeeze()
        SSIM_av_per_q_N = SSIM_vec.mean(axis=1)
        SSIM_std_per_q_N = SSIM_vec.std(axis=1)


        plt.plot(q_vec, SSIM_av_per_q_N, linestyle='solid', label=label_str,color=colorslist[r],
                     marker=markers[r])

        plt.fill_between(q_vec, (SSIM_av_per_q_N - SSIM_std_per_q_N/2), (SSIM_av_per_q_N + SSIM_std_per_q_N/2), color=colorslist[r],alpha=0.1)

        #print(f'R={R} SSIM ', SSIM_av_per_q_N)

        # compute the change between the cases of No-Compression (NC, q=101) to high compression (q=20)
        SSIM_diff = SSIM_av_per_q_N[q20_i] - SSIM_av_per_q_N[qNC_i]
        SSIM_increase = SSIM_diff / SSIM_av_per_q_N[qNC_i] * 100
        #print(f'{method_str}  R{R} SSIM increase from 20 to NC: {SSIM_increase}%')

        print(f'R={R} SSIM init {SSIM_av_per_q_N[-1]:.2f}; SSIM end {SSIM_av_per_q_N[0]:.2f}; rise {SSIM_increase}%')

        # # plot lines between NC and q20
        #if R==np.max(R_vec):
            # # plot horizontal line
            # plt.plot(np.array([100, 20]), (SSIM_av_per_q_N[qNC_i].item(), SSIM_av_per_q_N[qNC_i].item()), 'k--')
            #
            # # plot vertical line
            # plt.plot(np.array([20, 20]), np.array([SSIM_av_per_q_N[q20_i].item(), SSIM_av_per_q_N[qNC_i].item()]),
            #          'k--')

    plt.xlabel('JPEG quality', fontsize=20)
    plt.ylabel('SSIM', fontsize=20)
    ax = plt.gca()
    ax.set_xticks(q_vec[::-1])
    ax.set_xticklabels(x_ticks_labels, fontsize=14)
    ax.invert_xaxis()
    plt.locator_params(axis='y', nbins=4)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.title(method_str)
    if np.max(R_vec)==3:
        plt.ylim(0.88, 1)
        ax.set_yticks((0.9, 0.95, 1))
    elif np.max(R_vec)==4:
        plt.ylim(0.82, 1)
        ax.set_yticks((0.85, 0.9, 0.95, 1))

    elif np.max(R_vec)==6:
        plt.ylim(0.6, 1)


    if method_i == N_methods:
        ax.legend(fontsize=15, loc='lower right')
    plt.show()

    figname_SSIM = figs_path + f'/{method_str}_SSIM_stats'
    fig.savefig(fname=figname_SSIM)


