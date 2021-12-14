'''
This code is used for training MoDL on zero-padded data, for the results shown in figures 5 and 8a-b in the paper.

Before running this script you should update the following:
basic_data_folder - it should be the same as the output folder defined in the script /crime_2_jpeg/data_prep/jpeg_data_prep.py

(c) Efrat Shimron, UC Berkeley, 2021
'''

##############################################################################################
# Example - how to run this script from linux command line:
# python3 Train_xxxxx.py  --R 4 --pad_ratio 1 --gpu 0 --var_dens_flag 'strong'
###############################################################################################


# %matplotlib notebook
import os, sys
import logging
import numpy as np
import torch
import torch.nn as nn
import copy

import matplotlib.pyplot as plt
from subtle_data_crimes.crime_1_zero_padding.Fig4_pathology_example.DL.utils import complex_utils as cplx
from subtle_data_crimes.crime_1_zero_padding.Fig4_pathology_example.DL.utils.datasets import \
    create_data_loaders  # , calc_scaling_factor
from subtle_data_crimes.crime_1_zero_padding.Fig4_pathology_example.DL.MoDL_single import UnrolledModel
from subtle_data_crimes.functions import error_metrics
import argparse





def create_arg_parser():
    parser = argparse.ArgumentParser(description="Subtle inverse crimes - MoDL script")
    parser.add_argument('--pad_ratio', type=float, default=2, help='zero padding ratio')
    parser.add_argument('--R', type=int, default=4, help='Reduction Factor')
    parser.add_argument('--unrolls', type=int, default=6, help='Reduction Factor')
    parser.add_argument('--var_dens_flag', type=str, default='weak', help='variable density strong/weak')
    parser.add_argument('--gpu', type=int, default=0, help='GPU')

    return parser


if __name__ == '__main__':

    args = create_arg_parser().parse_args()

    if args.pad_ratio >=3:
        use_multiple_GPUs_flag = 1
    else:
        use_multiple_GPUs_flag = 0

    #use_multiple_GPUs_flag = 0

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
    #params.num_grad_steps = 6  # number of unrolls
    params.num_cg_steps = 8
    params.share_weights = True
    params.modl_lamda = 0.05
    params.lr = 0.0001
    params.weight_decay = 0
    params.lr_step_size = 500
    params.lr_gamma = 0.5
    #params.epoch = 70 # This was used for R4 runs
    params.epoch = 70
    params.num_grad_steps = args.unrolls
    params.R = args.R
    params.pad_ratio = args.pad_ratio # zero-padding ratio
    #params.sampling_flag = 'var_dens_1D'
    print('2D VAR DENS')
    params.sampling_flag = 'var_dens_2D'
    params.var_dens_flag = args.var_dens_flag
    params.NX_full_FOV = 640
    params.NY_full_FOV = 372
    # params.sampling_flag = 'random_uniform'

    im_type_str = 'blocks' # for training and validation we use blocks, but for test (inference) we use full images

    NX_block = 256
    NY_block = 134

    block_to_im_ratio = NX_block/640

    # calib is assumed to be 12 for NX=640
    calib_x = int(12 * block_to_im_ratio * params.pad_ratio)
    calib_y = int(12 * block_to_im_ratio * params.pad_ratio * (NY_block/NX_block))
    params.calib = np.array([calib_x, calib_y])

    params_val = copy.copy(params)


    FatSat_processed_data_folder = "/mikQNAP/NYU_knee_data/efrat/public_repo_check/zpad_FatSat_data/" # FatSatPD
    print('FatSatPD data is used!')

    FatSat_processed_data_folder = FatSat_processed_data_folder + '/'

    # path to train data
    data_type = 'train'
    params.data_path = FatSat_processed_data_folder + data_type + "/pad_" + str(
        int(100 * params.pad_ratio)) + "/" + im_type_str + "/"

    # path to val data
    data_type = 'val'
    params_val.data_path = FatSat_processed_data_folder + data_type + "/pad_" + str(
        int(100 * params.pad_ratio)) + "/" + im_type_str + "/"

    # create a directory for the current MoDL run
    run_foldername = 'R{}_pad_{}_unrolls_{}_{}_var_dens'.format(params.R, str(int(100 * params.pad_ratio)),
                                                                args.unrolls, args.var_dens_flag)


    print(params.data_path)

    print('num unrolls=',args.unrolls)
    print('gpu = ',args.gpu)
    print('pad_ratio=',params.pad_ratio)
    print('R=', params.R)
    print('var_dens_flag=',params.var_dens_flag)



    # Create data loader
    train_loader = create_data_loaders(params)
    val_loader = create_data_loaders(params_val)


    N_train_slices = len(train_loader.dataset) # assuming that batch_size = 1, N_train_slices = N_train_datasests
    N_val_slices = len(val_loader.dataset)  # assuming that batch_size = 1, N_train_slices = N_train_datasests
    print('N_train_slices=', N_train_slices)
    print('N_val_slices=',N_val_slices)


    # Create an unrolled model
    single_MoDL = UnrolledModel(params).to(device)


    # Data Parallelism - enables running on multiple GPUs
    if (torch.cuda.device_count()>1) & (use_multiple_GPUs_flag==1):
        print("Now using ", torch.cuda.device_count(), "GPUs!")
        single_MoDL = nn.DataParallel(single_MoDL, device_ids =  [0,1,2,3]) # the first index on the device_ids determines which GPU will be used as a staging area before scattering to the other GPUs
    else:
        print("Now using a single GPU")

    optimizer = build_optim(params, single_MoDL.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.lr_step_size, params.lr_gamma)
    criterion = nn.L1Loss()
    loss_train_data = list([])


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
            #torch.set_grad_enabled(True)
            #input, target, mask, target_no_JPEG = data
            input, target, mask = data

            # display and print the mask (before converting it to torch tensor)
            if (epoch == 0) & (iter == 0):
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
            if (torch.cuda.device_count()>1) & (use_multiple_GPUs_flag==1):
                input = input.to(f'cuda:{single_MoDL.device_ids[0]}')
                target = target.to(f'cuda:{single_MoDL.device_ids[0]}')
                mask = mask.to(f'cuda:{single_MoDL.device_ids[0]}')
            else:
                input = input.to(device)
                target = target.to(device)
                mask = mask.to(device)

            # forward pass
            im_out = single_MoDL(input.float(), mask=mask)

            # fig = plt.figure()
            # plt.imshow(single_MoDL.zf_im_4plot,cmap="gray")
            # plt.title('zf image - check the scale!!!')
            # plt.colorbar()
            # plt.show()
            # #filename = run_foldername + "/zf_image.png"
            # #fig.savefig(filename)

            # # debugging plot - validation
            # im_out_1_detached = im_out.detach()
            # im_out_1 = cplx.to_numpy(im_out_1_detached.cpu())[0, :, :]

            # fig = plt.figure()
            # plt.imshow(np.abs(im_out_1),cmap="gray")
            # plt.title('CHECK SCALE!!! im out - Training epcoh {} iter {}'.format(epoch,iter))
            # plt.colorbar()
            # plt.show()


            # calc training loss
            loss = criterion(im_out, target)

            loss = loss / num_accumulated_iters  # because the loss is accumulated for num_accumulated_iters, we divide the loss by the number of iters to average the accumulated loss gradients.

            # backward pass & update network parameters
            # optimizer.zero_grad()  # this part was moved above for gradient accumulation
            loss.backward()  # backward pass. note that gradients are accumulated as long as we don't do optimizer.zero_grad()

            if (iter + 1) % num_accumulated_iters == 0:
                # Do a SGD step once every num_accumulated_iters
                optimizer.step()
                optimizer.zero_grad()

            loss_train_data.append(loss.item())


            avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()

            # logging info
            if iter % 20 == 0:
                logging.info(
                    f'Epoch = [{epoch:3d}/{params.epoch:3d}] '
                    f'Iter = [{iter:4d}/{len(train_loader):4d}] '
                    f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g}'
                )

            if (iter == 0) & ((epoch <=10) | (epoch % 10 ==0 )):
                fig = plt.figure()
                loss_train_data_arr = np.asarray(loss_train_data)
                plt.plot(loss_train_data_arr)
                plt.xlabel('iter')
                plt.ylabel('loss')
                plt.title('training data loss - iter 0 , epoch {}'.format(epoch))
                plt.show()
                fig.savefig(run_foldername + '/loss_fig.png')

                print('loss fig saved')

            # debugging plots - show training example
            if (iter==0) & ((epoch <=10) | (epoch % 10 ==0 )):
                input_detached = input.detach()
                out_detached = im_out.detach()
                target_detached = target.detach()
                im_input = cplx.to_numpy(input_detached.cpu())[iter, :, :]
                im_out = cplx.to_numpy(out_detached.cpu())[iter, :, :]
                im_target = cplx.to_numpy(target.cpu())[iter, :, :]

                MoDL_err = error_metrics(np.abs(im_target), np.abs(im_out))
                MoDL_err.calc_NRMSE()
                MoDL_err.calc_SSIM()

                fig = plt.figure(figsize=(15,7))

                plt.subplot(1, 3, 1)
                plt.imshow(single_MoDL.zf_im_4plot, cmap="gray")
                plt.colorbar(shrink=0.5)
                plt.title('zf image')

                plt.subplot(1, 3, 2)
                plt.imshow(np.rot90(np.abs(im_out), 2), cmap="gray")
                plt.colorbar(shrink=0.5)
                plt.title('im_out MoDL - NRMSE {:0.3f}'.format(MoDL_err.NRMSE))
                # plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(np.rot90(np.abs(im_target), 2), cmap="gray")
                plt.colorbar(shrink=0.5)
                # plt.axis('off')
                plt.title('target')
                plt.suptitle('TRAINING example - iter {} epoch {} - check scale!!!'.format(iter,epoch))
                plt.show()
                fig.savefig(run_foldername + '/sanity_check_fig_{}.png'.format(monitoring_cnt))

                monitoring_cnt += 1

                print('training sanity fig saved')

                # -------------- run validation ---------------------
                # notice: validation requires changing the model's state using torch.no_grad() and model.eval()
                print('validation...  during training epoch {}'.format(epoch))

                with torch.no_grad():
                    for val_iter, val_data in enumerate(val_loader):

                        if val_iter == 0:  # run validation for a single exmple

                            val_input_batch, val_target_batch, val_mask_batch = val_data

                            # move data to GPU
                            if (torch.cuda.device_count() > 1) & (use_multiple_GPUs_flag == 1):
                                val_input_batch = val_input_batch.to(f'cuda:{single_MoDL.device_ids[0]}')
                                val_target_batch = val_target_batch.to(f'cuda:{single_MoDL.device_ids[0]}')
                                val_mask_batch = val_mask_batch.to(f'cuda:{single_MoDL.device_ids[0]}')
                            else:
                                val_input_batch = val_input_batch.to(device)
                                val_target_batch = val_target_batch.to(device)
                                val_mask_batch = val_mask_batch.to(device)

                            # forward pass - for the full batch
                            single_MoDL.eval()
                            val_out_batch = single_MoDL(val_input_batch.float(), mask=val_mask_batch)

                            for iii in range(1):
                                #print('iii=',iii)
                                val_im_input = cplx.to_numpy(val_input_batch.cpu())[iii, :, :]
                                val_im_target = cplx.to_numpy(val_target_batch.cpu())[iii, :, :]
                                val_im_out = cplx.to_numpy(val_out_batch.cpu())[iii, :, :]

                                print('-----------------------------')
                                print('val_im_target shape:',val_im_target.shape)
                                print('-----------------------------')

                                MoDL_val_err = error_metrics(np.abs(val_im_target), np.abs(val_im_out))
                                MoDL_val_err.calc_NRMSE()
                                MoDL_val_err.calc_SSIM()

                                fig = plt.figure(figsize=(15, 7))

                                plt.subplot(1, 3, 1)
                                plt.imshow(single_MoDL.zf_im_4plot, cmap="gray")
                                plt.colorbar(shrink=0.5)
                                plt.title('val zf image')

                                plt.subplot(1, 3, 2)
                                plt.imshow(np.rot90(np.abs(val_im_out), 2), cmap="gray")
                                plt.colorbar(shrink=0.5)
                                plt.title('val_im_out MoDL - NRMSE {:0.3f}'.format(MoDL_val_err.NRMSE))
                                # plt.axis('off')

                                plt.subplot(1, 3, 3)
                                plt.imshow(np.rot90(np.abs(val_im_target), 2), cmap="gray")
                                plt.colorbar(shrink=0.5)
                                # plt.axis('off')
                                plt.title('val_target')
                                plt.suptitle('VALIDATION example - training iter {} epoch {}'.format(iter, epoch))
                                plt.show()
                                fig.savefig(run_foldername + '/val_check_fig_{}.png'.format(monitoring_cnt))
                                print('validation sanity fig saved')


                # # ---- old plot ----
                # input_numpy = cplx.to_numpy(input.cpu())[0, :, :]
                #
                # # fig = plt.figure()
                # # plt.imshow(np.log(np.abs(input_numpy)),cmap="gray")
                # # plt.title('input k-space')
                # # plt.show()
                # # fig.savefig('check input kspace')
                #
                # # fig = plt.figure()
                # # plt.imshow(np.abs(mask[0,:,:,0])),cmap="gray")
                # # plt.show()
                # # fig.savefig(run_foldername + '/mask_{}.png'.format(monitoring_cnt))
                #
                # # extract a single image from the batch & convert it from a two-channel tensor (Re&Im) to a complex numpy array
                # im_target_1 = cplx.to_numpy(target.cpu())[0, :, :]
                # im_out_1_detached = im_out.detach()
                # im_out_1 = cplx.to_numpy(im_out_1_detached.cpu())[0, :, :]
                #
                # a = np.concatenate((im_target_1, im_out_1), axis=1)
                # s1 = im_target_1.shape[1]
                #
                # fig = plt.figure()
                # plt.subplot(1, 2, 1)
                # plt.imshow(np.rot90(np.abs(im_target_1),2), cmap="gray")
                # #plt.axis('off')
                # plt.colorbar(shrink=0.5)
                # plt.title('target')
                #
                # plt.subplot(1, 2, 2)
                # plt.imshow(np.rot90(np.abs(im_out_1),2), cmap="gray")
                # plt.colorbar(shrink=0.5)
                # #plt.axis('off')
                # plt.title('im out - training epoch {} iter {}'.format(epoch,iter) )
                # plt.suptitle('loss {:.4g}'.format(loss.item()))
                # plt.show()
                # fig.savefig(run_foldername + '/sanity_check_fig_{}.png'.format(monitoring_cnt))
                #
                # monitoring_cnt += 1
                #
                # print('monitoring fig saved')




        # Saving the model
        exp_dir = run_foldername + "/checkpoints/"
        torch.save(
            {
                'epoch': epoch,
                'params': params,
                'model': single_MoDL.state_dict(),
                'optimizer': optimizer.state_dict(),
                'exp_dir': exp_dir,
                'loss_train_data': loss_train_data,
            },
            f=os.path.join(exp_dir, 'model_%d.pt' % (epoch))
        )

        # save the loss
        def saveList(myList,filename):
            # the filename should mention the extension 'npy'
            np.save(filename,myList)
            print("Saved successfully!")

        saveList(loss_train_data,run_foldername + '/loss_train_data.npy')

        # # how to load the list loss_train_data into a numpy array:
        # def loadList(filename):
        #     # the filename should mention the extension 'npy'
        #     tempNumpyArray=np.load(filename)
        #     return tempNumpyArray.tolist()








