"""
This code is based on the code subsample_fastmri.py by Facebook, Inc. and its affiliates
The new additions include functions for variable-density sampling, which are based on Miki Lustig's l1-SPIRiT toolbox.

Efrat Shimron, UC Berkeley (2021)

"""

import numpy as np
import torch
from subtle_data_crimes.functions.sampling_funcs import gen_1D_var_dens_mask, gen_2D_var_dens_mask , genPDF,genSampling

class MaskFuncVarDens_1D:
    """
    This code produces a 1D variable-density mask of a given shape.
    MaskFuncVarDens_1D creates a sub-sampling mask
    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)
    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.
    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.
    """

    def __init__(self, calib,R,pad_ratio,var_dens_flag):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.
            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        #if len(center_fractions) != len(accelerations):
        #    raise ValueError('Number of center fractions should match number of accelerations')

        #self.center_fractions = center_fractions
        #self.accelerations = accelerations
        self.rng = np.random.RandomState()
        self.R = R
        #self.calib = calib
        self.calib = calib*pad_ratio
        self.var_dens_flag = var_dens_flag

    def __call__(self, shape, seed=None):

        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """


        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)
        num_cols = shape[-2]

        #choice = self.rng.randint(0, len(self.accelerations))
        #center_fraction = self.center_fractions[choice]
        #acceleration = self.accelerations[choice]

        # Create the mask
        # num_low_freqs = int(round(num_cols * center_fraction))
        # prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        #
        # # the next code produces a 1D mask (i.e. a vector of False/True values indicated columns to be sampled)
        # mask = self.rng.uniform(size=num_cols) < prob
        # pad = (num_cols - num_low_freqs + 1) // 2
        # mask[pad:pad + num_low_freqs] = True

        R = self.R
        calib = self.calib

        if R >= 8:
            poly_degree = 10
        elif R == 7:
            poly_degree = 8
        elif R == 6:
            if self.var_dens_flag=='weak':
                poly_degree = 12
            elif self.var_dens_flag=='strong':
                poly_degree = 6
        elif R == 5:
            if self.var_dens_flag == 'weak':
                poly_degree = 5
            #elif self.var_dens_flag == 'strong':
                #TODO: implement this
        elif R == 4:
            if self.var_dens_flag == 'weak':
                poly_degree = 10
            elif self.var_dens_flag == 'strong':
                poly_degree = 4
        elif R == 3:
            poly_degree = 4
        elif R == 2:
            poly_degree = 4

        # # create a 1D pdf
        # pdf = self.genPDF(num_cols, poly_degree, 1 / R)
        # mask = self.genSampling(pdf, iter=10, tol=60, calib=calib)

        ## create a 1D pdf
        #pdf = genPDF(np.array([num_cols]), poly_degree, 1 / R)
        #mask = genSampling(pdf, iter=50, tol=20,calib=calib )

        mask, pdf = gen_1D_var_dens_mask()

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = mask.reshape(*mask_shape).astype(np.float32)

        return mask



class MaskFuncVarDens_2D:
    """
        This code produces a 2D variable-density mask.
        It is highly similar to the 1D case above.
    """

    def __init__(self, calib, R, pad_ratio, var_dens_flag):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.
            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        # if len(center_fractions) != len(accelerations):
        #    raise ValueError('Number of center fractions should match number of accelerations')

        # self.center_fractions = center_fractions
        # self.accelerations = accelerations
        self.rng = np.random.RandomState()
        self.R = R
        # self.calib = calib
        self.calib = calib
        self.var_dens_flag = var_dens_flag

    def __call__(self, shape, seed=None):

        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. In the 1D case, samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """

        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)
        num_rows = shape[-3]
        num_cols = shape[-2]


        imSize = np.array([num_rows,num_cols])
        mask, pdf, poly_degree_new = gen_2D_var_dens_mask(self.R, imSize, self.var_dens_flag,calib = self.calib)

        # # debugging plot
        # fig = plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(np.abs(pdf),cmap="gray")
        #
        # plt.subplot(1,2,2)
        # plt.imshow(np.abs(mask),cmap="gray")
        # plt.show()
        # filename = 'pdf_example_NX_{}_NY_{}'.format(num_rows,num_cols)
        # fig.savefig(filename)

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask_shape[-3] = num_rows
        mask_full_shape = mask.reshape(*mask_shape).astype(np.float32)

        return mask_full_shape
