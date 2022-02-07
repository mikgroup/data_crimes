'''
This script loads the results of CS, DictL and DL, with:
#    pad_ratio = 1, weak VD
#    pad_ratio = 2, weak VD
#    pad_ratio = 1, strong VD
#    pad_ratio = 2, strong VD
# and prepares figures for Fig 4 in the paper.

# (c) Efrat Shimron, UC Berkeley, 2021
'''

import os

import matplotlib.pyplot as plt
import numpy as np

from subtle_data_crimes.functions import error_metrics

R = 4

pad_ratio_vec = np.array([1, 2])
sampling_type_vec = np.array([1])  # 1 = weak VD, 2 = strong VD
methods_list = ['CS', 'DictL', 'DL']
data_type = 'pathology_1'

zones_vec = np.array([1, 2])  # np.array([1,2])

for method_i, method_str in enumerate(methods_list):
    print('==================================================')
    print(f'                   {method_str}                  ')
    print('==================================================')

    for pad_i, pad_ratio in enumerate(pad_ratio_vec):

        pad_ratio_str = int(pad_ratio_vec[pad_i])

        for j in range(sampling_type_vec.shape[0]):

            if sampling_type_vec[j] == 0:  # random uniform
                samp_type = 'random'
            elif sampling_type_vec[j] == 1:  # weak variable-density
                samp_type = 'weak'
            elif sampling_type_vec[j] == 2:  # strong variable-density
                samp_type = 'strong'

            print('------------------------------------------------------------------------------')
            print(f'                   {samp_type} VD & pad {pad_ratio_str}            ')
            print('-------------------------------------------------------------------------------')

            results_dir = method_str + '/' + data_type + f'_results_R{R}/'
            gold_filename = results_dir + 'gold_dict.npy'
            rec_filename = results_dir + method_str + '_dict.npy'

            gold_container = np.load(gold_filename, allow_pickle=True)
            rec_container = np.load(rec_filename, allow_pickle=True)

            rec_gold_rotated = gold_container.item()[(pad_ratio, samp_type)]
            rec_rotated = rec_container.item()[(pad_ratio, samp_type)]

            # compute NRMSE & SSIM
            A = error_metrics(rec_gold_rotated, rec_rotated)
            A.calc_NRMSE()
            A.calc_SSIM()
            print(f'{method_str} rec; NRMSE={A.NRMSE:.4f}')

            cmax = np.max([np.abs(rec_gold_rotated), np.abs(rec_rotated)])

            # # display full-size images
            # fig = plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(rec_gold_rotated, cmap="gray")
            # plt.title('rec_gold')
            # plt.clim(0,cmax)
            # plt.colorbar(shrink=0.25)
            #
            # plt.subplot(1,2,2)
            # plt.imshow(rec_rotated,cmap="gray")
            # plt.title(f'CS NRMSE {A.NRMSE:.3f}')
            # plt.clim(0, cmax)
            # plt.colorbar(shrink=0.25)
            # plt.suptitle(f'{data_type} data; R={R}; pad_ratio={pad_ratio}; {samp_type} VD samp; scan {t}; slice {ns}')
            # plt.show()
            # figname = figs_folder + f'/slice{ns}_pad_{pad_ratio}_{samp_type}.png'
            # fig.savefig(figname)

            if (method_i == 0) & (pad_i == 0) & (j == 0):
                fig = plt.figure()
                plt.imshow(rec_gold_rotated, cmap="gray")
                # plt.axis('off')
                plt.clim(0, cmax)
                # plt.title(f'gold pad{pad_ratio_str}')
                plt.show()
                # figname = figs_folder + f'/gold_full_size.eps'
                # fig.savefig(figname, dpi=1000)

            for z_i in range(zones_vec.shape[0]):
                if zones_vec[z_i] == 1:
                    # prepare zoomed-in figures for the paper
                    # zoom-in coordinates for pathology 1
                    x1 = 335
                    x2 = 380
                    y1 = 210
                    y2 = 300
                    # scale the zoom-in coordinates to fit changing image size
                    x1s = int(x1 * pad_ratio)
                    x2s = int(x2 * pad_ratio)
                    y1s = int(y1 * pad_ratio)
                    y2s = int(y2 * pad_ratio)
                    zone_str = f'zone_1'

                elif zones_vec[z_i] == 2:
                    x1 = 325
                    x2 = 370
                    y1 = 70
                    y2 = 160
                    # scale the zoom-in coordinates to fit changing image size
                    x1s = int(x1 * pad_ratio)
                    x2s = int(x2 * pad_ratio)
                    y1s = int(y1 * pad_ratio)
                    y2s = int(y2 * pad_ratio)
                    zone_str = f'zone_2'

                figs_folder = 'Fig4_' + data_type + '_' + zone_str
                if not os.path.exists(figs_folder):
                    os.makedirs(figs_folder)

                # # rec CS zoomed - png figure
                # fig = plt.figure()
                # plt.imshow(rec_rotated[x1s:x2s, y1s:y2s], cmap="gray")
                # plt.axis('off')
                # plt.clim(0, cmax)
                # plt.show()
                # figname = figs_folder + f'/CS_pad_x{pad_ratio_str}_{samp_type}_VD_zoomed'
                # fig.savefig(figname, dpi=1000)

                # rec CS zoomed - eps figure
                fig = plt.figure()
                plt.imshow(rec_rotated[x1s:x2s, y1s:y2s], cmap="gray")
                plt.axis('off')
                plt.clim(0, cmax)
                plt.title(f'{method_str} pad {pad_ratio_str} {samp_type} VD')
                plt.show()
                figname = figs_folder + f'/{method_str}_pad_x{pad_ratio_str}_{samp_type}_VD_zoomed.eps'
                fig.savefig(figname, format='eps', dpi=1000)

                if (method_i == 0) & (pad_ratio == 2) & (j == 0):
                    # # gold standard zoomed - png figure
                    # fig = plt.figure()
                    # plt.imshow(rec_gold_rotated[x1s:x2s, y1s:y2s], cmap="gray")
                    # plt.axis('off')
                    # plt.clim(0, cmax)
                    # plt.show()
                    # figname = figs_folder + f'/gold_pad_x{pad_ratio_str}_zoomed.png'
                    # fig.savefig(figname, dpi=1000)

                    # gold standard zoomed - eps figure
                    fig = plt.figure()
                    plt.imshow(rec_gold_rotated[x1s:x2s, y1s:y2s], cmap="gray")
                    plt.axis('off')
                    plt.clim(0, cmax)
                    plt.title(f'gold pad {pad_ratio_str}')
                    plt.show()
                    figname = figs_folder + f'/gold_pad_x{pad_ratio_str}_zoomed.eps'
                    fig.savefig(figname, format='eps', dpi=1000)

                    if z_i == 0:
                        # # gold standard full-size png figure
                        # fig = plt.figure()
                        # plt.imshow(rec_gold_rotated, cmap="gray")
                        # plt.axis('off')
                        # plt.clim(0, cmax)
                        # plt.show()
                        # figname = figs_folder + f'/gold_full_size'
                        # fig.savefig(figname, dpi=1000)

                        # gold standard full-size .png figure
                        fig = plt.figure()
                        plt.imshow(rec_gold_rotated, cmap="gray")
                        # plt.axis('off')
                        plt.clim(0, cmax)
                        # plt.title(f'gold pad{pad_ratio_str}')
                        plt.show()
                        figname = figs_folder + f'/gold_full_size.eps'
                        fig.savefig(figname, dpi=1000)

