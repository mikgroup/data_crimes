'''
This code is used for training MoDL on JPEG-compressed data, for the results shown in figures 6, 7 and 8c in the paper.

Before running this script you should update the following:
basic_data_folder - it should be the same as the output folder defined in the script /crime_2_jpeg/data_prep/jpeg_data_prep.py

(c) Efrat Shimron, UC Berkeley, 2021
'''


import os, sys
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import complex_utils as cplx
from utils.datasets import create_data_loaders
from MoDL_single import UnrolledModel
import argparse





def create_arg_parser():
    parser = argparse.ArgumentParser(description="Subtle inverse crimes - MoDL script")
    parser.add_argument('--q', type=int, default=100, help='JPEG compression param')
    parser.add_argument('--R', type=int, default=4, help='Reduction Factor')
    parser.add_argument('--gpu', type=int, default=1, help='GPU')

    return parser


if __name__ == '__main__':

    args = create_arg_parser().parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    def build_optim(args, params):
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


    # Hyper parameters
    params = Namespace()
    params.batch_size = 1
    params.num_grad_steps = 6  # number of unrolls
    params.num_cg_steps = 8
    params.share_weights = True
    params.modl_lamda = 0.05
    params.lr = 0.0001
    params.weight_decay = 0
    params.lr_step_size = 500
    params.lr_gamma = 0.5
    #params.epoch = 70 # This was used for R4 runs
    params.epoch = 70
    #params.calib = 24
    params.R = args.R
    params.q = args.q  # JPEG compression parameter
    params.shuffle_flag = 'True'  # should be True for training, False for testing

    # image dimensions
    params.NX = 640
    params.NY = 372

    # calib is assumed to be 12 for NX=640
    calib_x = int(12)
    calib_y = int(12*params.NY/params.NX)
    params.calib = np.array([calib_x, calib_y])

    # params.sampling_flag = 'random_uniform'
    #params.sampling_flag = 'var_dens_1D'
    print('2D VAR DENS')
    params.sampling_flag = 'var_dens_2D'
    params.var_dens_flag = 'weak' # 'weak' / 'strong'
    im_type_str = 'full_im'

    # where to save the output files:
    basic_out_folder = "/mikQNAP/NYU_knee_data/multicoil_efrat/5_JPEG_compressed_data"

    data_type = 'train'

    basic_out_folder = basic_out_folder + '/'
    params.data_path = basic_out_folder + data_type + "/q" + str(params.q) + "/" + im_type_str + "/"
    run_foldername = 'R{}_q{}'.format(params.R, params.q)

    print(f'CHECK THIS: params.data_path= {params.data_path}')


    print('q=',params.q)
    print('R=', params.R)

    # Create data loader
    train_loader = create_data_loaders(params)

    N_train_scans = len(train_loader.dataset)
    print('N_train_scans=',N_train_scans)

    # Create an unrolled model
    single_MoDL = UnrolledModel(params).to(device)

    single_MoDL.display_zf_image_flag = 1  # display & save the zero-filled recon (initial guess) only in the first iter


    optimizer = build_optim(params, single_MoDL.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.lr_step_size, params.lr_gamma)
    criterion = nn.L1Loss()
    loss_all = list([])

    # create sub-directory for saving checkpoints:
    if not os.path.exists(run_foldername + "/checkpoints"):
        os.makedirs(run_foldername + "/checkpoints")


    optimizer.zero_grad()
    num_accumulated_iters = 20  # accumulate iters (this enables using batch_size=1 and fitting large networks on a single GPU)

    # Training

    monitoring_cnt = 0
    for epoch in range(params.epoch):
        single_MoDL.train()
        avg_loss = 0.

        for iter, data in enumerate(train_loader):
            #input, target, mask, target_no_JPEG = data
            input, target, mask = data

            # display and print the mask (before converting it to torch tensor)
            if (epoch == 0) & (iter <= 5):
                # print('mask shape:',mask.shape)
                mask_squeezed = mask[0, :, :, 0].squeeze()
                fig = plt.figure()
                plt.imshow(mask_squeezed, cmap="gray")
                plt.title(params.sampling_flag + ' epoch 0, iter {}'.format(iter))
                plt.show()
                fig.savefig(run_foldername + '/mask_iter{}.png'.format(iter))

            # # move data to GPU
            # input = input.to(device)
            # target = target.to(device)
            # mask = mask.to(device)

            # move data to GPU
            input = input.to(device)
            target = target.to(device)
            mask = mask.to(device)

            # forward pass
            im_out = single_MoDL(input.float(), mask=mask)

            # calc loss
            loss = criterion(im_out, target)

            loss = loss / num_accumulated_iters  # because the loss is accumulated for num_accumulated_iters, we divide the loss by the number of iters to average the accumulated loss gradients.

            # backward pass & update network parameters
            loss.backward()  # backward pass. note that gradients are accumulated, so optimizer.zero_grad() appears somewhere else.

            if (iter + 1) % num_accumulated_iters == 0:
                # Do a SGD step once every num_accumulated_iters
                optimizer.step()
                optimizer.zero_grad()

            loss_all.append(loss.item())


            avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()

            if single_MoDL.display_zf_image_flag != 0:
                single_MoDL.display_zf_image_flag = 0

            # logging info
            if iter % 20 == 0:
                logging.info(
                    f'Epoch = [{epoch:3d}/{params.epoch:3d}] '
                    f'Iter = [{iter:4d}/{len(train_loader):4d}] '
                    f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g}'
                )

            # display
            if (iter == 0) & ((epoch <=5) | (epoch % 10 ==0 )):
                # extract a single image from the batch & convert it from a two-channel tensor (Re&Im) to a complex numpy array
                im_target_1 = cplx.to_numpy(target.cpu())[0, :, :]
                im_out_1_detached = im_out.detach()
                im_out_1 = cplx.to_numpy(im_out_1_detached.cpu())[0, :, :]

                a = np.concatenate((im_target_1, im_out_1), axis=1)
                s1 = im_target_1.shape[1]

                fig = plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(np.rot90(np.abs(im_target_1),2), cmap="gray")
                plt.axis('off')
                plt.colorbar(shrink=0.5)
                plt.title('target')

                plt.subplot(1, 2, 2)
                plt.imshow(np.rot90(np.abs(im_out_1),2), cmap="gray")
                plt.colorbar(shrink=0.5)
                plt.axis('off')
                plt.title('out - training epoch {} iter {}'.format(epoch,iter) )
                plt.suptitle('loss {:.4g}'.format(loss.item()))
                plt.show()
                fig.savefig(run_foldername + '/sanity_check_fig_{}.png'.format(monitoring_cnt))
                fig.savefig('monitoring_fig_{}.png'.format(monitoring_cnt))

                monitoring_cnt += 1

                print('monitoring fig saved')

                fig = plt.figure()
                loss_all_arr = np.asarray(loss_all)
                plt.plot(loss_all_arr)
                plt.xlabel('iter')
                plt.ylabel('loss')
                plt.title('loss - fig updated during the run')
                plt.show()
                fig.savefig(run_foldername + '/loss_fig_epoch{}_iter{}.png'.format(epoch,iter))


                print('loss fig saved')


        # Saving the model
        exp_dir = run_foldername + "/checkpoints/"
        torch.save(
            {
                'epoch': epoch,
                'params': params,
                'model': single_MoDL.state_dict(),
                'optimizer': optimizer.state_dict(),
                'exp_dir': exp_dir,
                'loss_all': loss_all,
            },
            f=os.path.join(exp_dir, 'model_%d.pt' % (epoch))
        )

        # save the loss
        def saveList(myList,filename):
            # the filename should mention the extension 'npy'
            np.save(filename,myList)
            print("Saved successfully!")

        saveList(loss_all,run_foldername + '/loss_all.npy')








