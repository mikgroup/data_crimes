'''
This script creates the graph displayed in Figure 2b of the Subtle Data Crimes paper.

Before running this script, please run the script fig_2b_experiments.py which computes the effective sampling rates
and saves the results in a file named R_eff_results_R6.npz. The current script loads that files and creates the figure.

(c) Efrat Shimron, UC Berkeley, 2021.
'''

import numpy as np
import matplotlib.pyplot as plt


R = np.array([6])
filename = 'R_eff_results_R{}.npz'.format(R)

container = np.load(filename)
R_eff_vs_pad_and_poly = container['R_eff_vs_pad_and_poly']
R_eff_vs_pad_and_poly_av = container['R_eff_vs_pad_and_poly_av']
pad_ratio_vec = container['pad_ratio_vec']
poly_degree_vec = container['poly_degree_vec']



############################# prep for plot ###############################
# prepare x ticks labels for the NRMSE and SSIM graphs
x = pad_ratio_vec
x_ticks_labels = []
for i in range(pad_ratio_vec.shape[0]):
    x_ticks_labels.append('x{}'.format(pad_ratio_vec[i]))


y_ticks = np.array([2,4,6])
y_ticks_labels = np.array([2,4,6])


styl_list=['-','--','-.',':']


################################### plot R_eff #############################
mycolor =['k','b','r','g']

fig = plt.figure()
for j, poly_degree in enumerate(poly_degree_vec):
    if poly_degree_vec[j]==1000:
        label='random uniform'
    elif poly_degree_vec[j]==10:
        label='weak var dens'
    elif poly_degree_vec[j]<=6:
        label = 'strong var dens'

    styl=styl_list[j]

    plt.plot(pad_ratio_vec, R_eff_vs_pad_and_poly_av[j, :], marker='o', markersize=7.50, label=label, linewidth=2,color=mycolor[j])

    plt.text(3.15,0.97*R_eff_vs_pad_and_poly_av[j,-1],label,fontsize=18)
plt.xlabel('zero padding',fontsize=18)
plt.ylabel('R_eff',fontsize=18,rotation='horizontal',labelpad=35)
plt.ylim((0,6.5))
plt.xlim((0.8,4.8))
ax = plt.gca()
ax.set_xticks(x)
ax.set_yticks(y_ticks)
ax.set_xticklabels(x_ticks_labels, fontsize=18)
ax.set_yticklabels(y_ticks_labels, fontsize=18)
plt.show()
fig.savefig('R_eff_vs_zero_pad_R{}_4paper'.format(R))



################################## plot effective sampling rate ###########################

y_ticks = np.array([20,40,60])
y_ticks_labels = ('20%','40%','60%')

fig = plt.figure()
for j, poly_degree in enumerate(poly_degree_vec):
    if poly_degree_vec[j]==1000:
        label='random uniform'
    elif poly_degree_vec[j]==10:
        label='weak var dens'
    elif poly_degree_vec[j]<=6:
        label = 'strong var dens'

    print('*** {} ***'.format(label))
    print('100*1/R=',100*1/R_eff_vs_pad_and_poly_av[j,:])
    styl=styl_list[j]
    plt.plot(pad_ratio_vec, 100*1/R_eff_vs_pad_and_poly_av[j, :], marker='o', markersize=7.50, label=label, linewidth=2,color=mycolor[j])

    plt.text(3.15,97*(1/R_eff_vs_pad_and_poly_av[j,-1]),label,fontsize=14,fontname="Times New Roman")
plt.xlabel('zero padding',fontsize=14,fontname="Times New Roman")
plt.ylim((0,65))
plt.xlim((0.8,4.6))
ax = plt.gca()
ax.set_xticks(x)
ax.set_yticks(y_ticks)
ax.set_xticklabels(x_ticks_labels, fontsize=14,fontname="Times New Roman")
ax.set_yticklabels(y_ticks_labels, fontsize=14,fontname="Times New Roman")
plt.show()
# fig.savefig('Eff_samp_vs_pad_4paper'.format(R)) # png figure
fig.savefig(fname=f'Eff_samp_vs_pad_4paper.eps', forrmat='eps', dpi=1000)


