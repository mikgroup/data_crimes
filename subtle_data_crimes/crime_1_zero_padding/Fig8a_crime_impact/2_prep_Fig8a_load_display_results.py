'''
This code loads the results and prepares Fig8a.

(c) Efrat Shimron, UC Berkeley, 2021
'''

import os

import matplotlib.pyplot as plt
import numpy as np

R = 4

pad_ratio_vec = np.array([1, 2])

print('============================================================================== ')
print('                        crime impact estimation                   ')
print('=============================================================================== ')

pad_ratio_test = np.array([1])

samp_type = 'weak'
method_str = 'DL'

# data_type = 'pathology_1'
data_type = 'pathology_2'

zones_vec = np.array([1, 2])  # np.array([1,2])

# load results
results_dir = data_type + f'_results_R{R}/'
gold_filename = results_dir + 'gold_dict.npy'
rec_filename = results_dir + method_str + '_dict.npy'

gold_container = np.load(gold_filename, allow_pickle=True)
rec_container = np.load(rec_filename, allow_pickle=True)

# display results per training pad ratio

for pad_i, pad_ratio in enumerate(pad_ratio_vec):  # training pad ratio; the test pad ratio was always 2

    pad_ratio_str = int(pad_ratio_vec[pad_i])

    rec_gold_rotated = gold_container.item()[(pad_ratio, samp_type)]
    rec_rotated = rec_container.item()[(pad_ratio, samp_type)]

    if pad_i == 0:
        cmax = np.max([np.abs(rec_gold_rotated), np.abs(rec_rotated)])

    # display full-FOV rec
    fig = plt.figure()
    plt.imshow(rec_rotated, cmap="gray")
    # plt.axis('off')
    plt.clim(0, cmax)
    plt.title(f'rec - train pad {pad_ratio_str}')
    plt.show()

    # display zoomed-in images
    for z_i in range(zones_vec.shape[0]):
        zone_str = f'zone_{zones_vec[z_i]}'
        figs_folder = f'Fig8a_R{R}' + data_type + '_' + zone_str
        if not os.path.exists(figs_folder):
            os.makedirs(figs_folder)

        if data_type == 'pathology_1':
            if zones_vec[z_i] == 1:
                # prepare zoomed-in figures for the paper
                # zoom-in coordinates for pathology 1
                x1 = 335
                x2 = 380
                y1 = 210
                y2 = 300
                # scale the zoom-in coordinates to fit changing image size
                x1s = int(x1 * pad_ratio_test)
                x2s = int(x2 * pad_ratio_test)
                y1s = int(y1 * pad_ratio_test)
                y2s = int(y2 * pad_ratio_test)

            elif zones_vec[z_i] == 2:
                x1 = 325
                x2 = 370
                y1 = 70
                y2 = 160
                # scale the zoom-in coordinates to fit changing image size
                x1s = int(x1 * pad_ratio_test)
                x2s = int(x2 * pad_ratio_test)
                y1s = int(y1 * pad_ratio_test)
                y2s = int(y2 * pad_ratio_test)
        elif data_type == 'pathology_2':
            if zones_vec[z_i] == 1:
                x1 = 310
                x2 = 350
                y1 = 200
                y2 = 280
                # scale the zoom-in coordinates to fit changing image size
                x1s = int(x1 * pad_ratio_test)
                x2s = int(x2 * pad_ratio_test)
                y1s = int(y1 * pad_ratio_test)
                y2s = int(y2 * pad_ratio_test)

            elif zones_vec[z_i] == 2:
                x1 = 310
                x2 = 360
                y1 = 90
                y2 = 190
                # scale the zoom-in coordinates to fit changing image size
                x1s = int(x1 * pad_ratio_test)
                x2s = int(x2 * pad_ratio_test)
                y1s = int(y1 * pad_ratio_test)
                y2s = int(y2 * pad_ratio_test)

        # rec zoomed - png
        fig = plt.figure()
        plt.imshow(rec_rotated[x1s:x2s, y1s:y2s], cmap="gray")
        plt.axis('off')
        plt.clim(0, cmax)
        plt.title(
            f'{method_str} train on pad {pad_ratio_str} & test on {str(int(pad_ratio_test))} - zone {zones_vec[z_i]}')
        plt.show()
        figname = figs_folder + f'/{method_str}_training_pad_x{pad_ratio_str}_{samp_type}_VD_zone_{zones_vec[z_i]}_zoomed'
        fig.savefig(figname, dpi=1000)

        # rec zoomed - eps figure
        fig = plt.figure()
        plt.imshow(rec_rotated[x1s:x2s, y1s:y2s], cmap="gray")
        plt.axis('off')
        plt.clim(0, cmax)
        plt.title(
            f'{method_str} train on pad {pad_ratio_str} & test on {str(int(pad_ratio_test))} - zone {zones_vec[z_i]}')
        plt.show()
        figname = figs_folder + f'/{method_str}_training_pad_x{pad_ratio_str}_{samp_type}_VD_zone_{zones_vec[z_i]}_zoomed.eps'
        fig.savefig(figname, format='eps', dpi=1000)
