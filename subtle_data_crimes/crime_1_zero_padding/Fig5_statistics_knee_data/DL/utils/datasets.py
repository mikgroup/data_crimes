"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import pathlib
import random
import torch

import h5py
from torch.utils.data import Dataset, DataLoader
import sigpy as sp
import matplotlib.pyplot as plt

from subtle_data_crimes.crime_1_zero_padding.Fig5_statistics_knee_data.DL.utils.subsample_fastmri import MaskFunc
from subtle_data_crimes.crime_1_zero_padding.Fig5_statistics_knee_data.DL.utils.subsample_var_dens import \
    MaskFuncVarDens_1D,MaskFuncVarDens_2D
from subtle_data_crimes.crime_1_zero_padding.Fig5_statistics_knee_data.DL.utils import complex_utils as cplx



# def calc_scaling_factor(kspace):
#     im_lowres = abs(sp.ifft(sp.resize(sp.resize(kspace, (640, 24)), (640, 372))))
#     magnitude_vals = im_lowres.reshape(-1)
#     k = int(round(0.05 * magnitude_vals.shape[0]))
#     scale = magnitude_vals[magnitude_vals.argsort()[::-1][k]]
#     return scale


class SliceData(Dataset):
    """
    A generic PyTorch Dataset class that provides access to 2D MR image slices.
    """

    def __init__(self, root, transform, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """

        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(num_slices)]
            # if num_slices==1:
            #     self.examples += [(fname, slice) for slice in range(1)]
            # else:
            #     # this code throws away 8 slices on each side of the scan - this was moved to the data preparation code
            #     self.examples += [(fname, slice+8) for slice in range(num_slices-16)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data['reconstruction'][slice]
            return self.transform(kspace,target,slice)


class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.
    """

    def __init__(self, mask_func, args, use_seed=False):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.rng = np.random.RandomState()
        self.sampling_flag  = args.sampling_flag
        self.NX_full_FOV = args.NX_full_FOV
        self.NY_full_FOV = args.NY_full_FOV


    def __call__(self, kspace, target, slice):
        # Note: the image data is assumed to be normalized to the range [0,1] (for magnitude data), and hence k-space is assumed to be of such data.

        # Convert everything from numpy arrays to tensors
        kspace_torch = cplx.to_tensor(kspace).float()
        target_torch = cplx.to_tensor(target).float()
        # print('kspace shape: ',kspace.shape)
        # mask_slice = np.ones((640, 372))
        mask_slice = np.ones((kspace.shape[0], kspace.shape[1]))
        # print('mask_slice.shape: ',mask_slice.shape)

        # call mask_func
        if self.sampling_flag == 'random_uniform_1D':
            mk1 = self.mask_func((1, 1, kspace.shape[1], 2))[0, 0, :, 0]
            knee_masks = mask_slice * mk1
            mask_torch = torch.tensor(knee_masks[..., None]).float()
            kspace_torch = kspace_torch * mask_torch

        elif self.sampling_flag == 'var_dens_1D':
            mk1 = self.mask_func((1, 1, kspace.shape[1], 2))[0, 0, :, 0]
            knee_masks = mask_slice * mk1
            mask_torch = torch.tensor(knee_masks[..., None]).float()
            kspace_torch = kspace_torch * mask_torch

        elif self.sampling_flag == 'var_dens_2D':
            mk1 = self.mask_func((1, kspace.shape[0], kspace.shape[1], 2))[0, :, :, 0]
            knee_masks = mask_slice * mk1
            mask_torch = torch.tensor(knee_masks[..., None]).float()
            kspace_torch = kspace_torch * mask_torch


        return kspace_torch, target_torch, mask_torch


def create_datasets(args):
    # Generate undersampling mask

    calib = args.calib
    R = args.R
    pad_ratio = args.pad_ratio
    var_dens_flag = args.var_dens_flag

    if args.sampling_flag == 'random_uniform_1D':
        #train_mask = MaskFunc([calib / 372], [R])  # random-uniform mask
        train_mask = MaskFunc(calib, [R])  # random-uniform mask

    elif args.sampling_flag == 'var_dens_1D':
        train_mask = MaskFuncVarDens_1D(calib, R, pad_ratio,var_dens_flag)  # variable-density mask

    elif args.sampling_flag == 'var_dens_2D':
        train_mask = MaskFuncVarDens_2D(calib, R, pad_ratio, var_dens_flag)  # variable-density mask

        # # Reshape the mask
        # mask_shape = [1 for _ in shape]
        # mask_shape[-2] = num_cols
        # mask_shape[-3] = num_rows
        # mask_full_shape = mask.reshape(*mask_shape).astype(np.float32)



    # print('train_mask shape:',train_mask.shape)

    # Explanation from utils.datasets.SliceData:
    # SliceData is a generic PyTorch Dataset class that provides access to 2D MR image slices.
    # Its inputs args are:
    #    root (pathlib.Path): Path to the dataset.
    #    transform (callable): A callable object that pre-processes the raw data into
    #           appropriate form. The transform function should take 'kspace', 'target',
    #           attributes', 'filename', and 'slice' as inputs. 'target' may be null for test data.
    #    sample_rate (float, optional): A float between 0 and 1. This controls what fraction
    #                 of the volumes should be loaded.

    train_data = SliceData(
        root=str(args.data_path),
        transform=DataTransform(train_mask, args),
        sample_rate=1
    )
    return train_data


def create_data_loaders(args,shuffle_flag = True):
    train_data = create_datasets(args)
    # print('train data shape:', train_data.shape)
    print('type(train data):', type(train_data))
    #     print(train_data[0])

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        #shuffle=True,
        shuffle = shuffle_flag,
        num_workers=32,
        pin_memory=True,
    )
    return train_loader



