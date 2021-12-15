"""
MoDL for single channel MRI 
by Ke Wang (kewang@berkeley.edu), 2020.

"""


import torch
from torch import nn
import subtle_data_crimes.crime_2_jpeg.Fig7.DL.utils.complex_utils as cplx
from subtle_data_crimes.crime_2_jpeg.Fig7.DL.utils.transforms import SenseModel,SenseModel_single
from subtle_data_crimes.crime_2_jpeg.Fig7.DL.unet.unet_model import UNet
from subtle_data_crimes.crime_2_jpeg.Fig7.DL.utils.flare_utils import ConjGrad

class Operator(torch.nn.Module):
    def __init__(self, A):
        super(Operator, self).__init__()
        self.operator = A

    def forward(self, x):
        return self.operator(x)

    def adjoint(self, x):
        return self.operator(x, adjoint=True)

    def normal(self, x):
        out = self.adjoint(self.forward(x))
        return out

class UnrolledModel(nn.Module):
    """
    PyTorch implementation of Unrolled Compressed Sensing.

    Implementation is based on:
        CM Sandino, et al. "DL-ESPIRiT: Accelerating 2D cardiac cine 
        beyond compressed sensing" arXiv:1911.05845 [eess.SP]
    """

    def __init__(self, params):
        """
        Args:
            params (dict): Dictionary containing network parameters
        """
        super().__init__()

        # Extract network parameters
        self.num_grad_steps = params.num_grad_steps 
        self.num_cg_steps = params.num_cg_steps
        self.share_weights = params.share_weights
        self.modl_lamda = params.modl_lamda

        # Declare ResNets and RNNs for each unrolled iteration
        if self.share_weights:
            print("shared weights")
            self.resnets = nn.ModuleList([UNet(2,2)] * self.num_grad_steps)
        else:
            print("No shared weights")
            self.resnets = nn.ModuleList([UNet(2,2) for i in range(self.num_grad_steps)])


    def forward(self, kspace, init_image=None, mask=None):
        """
        Args:
            kspace (torch.Tensor): Input tensor of shape [batch_size, height, width, time, num_coils, 2]
            maps (torch.Tensor): Input tensor of shape   [batch_size, height, width,    1, num_coils, num_emaps, 2]
            mask (torch.Tensor): Input tensor of shape   [batch_size, height, width, time, 1, 1]

        Returns:
            (torch.Tensor): Output tensor of shape       [batch_size, height, width, time, num_emaps, 2]
        """
        if mask is None:
            mask = cplx.get_mask(kspace)
        kspace *= mask

        # Get data dimensions
        dims = tuple(kspace.size())

        # Declare signal model
        A = SenseModel_single(weights=mask)
        Sense = Operator(A)
        # Compute zero-filled image reconstruction

        zf_image = Sense.adjoint(kspace)
        image = zf_image
        
        # Begin unrolled proximal gradient descent
        for resnet in self.resnets:
            # DC update
            image = image.permute(0,3,1,2)
            image = resnet(image)
            image = image.permute(0,2,3,1)
            rhs = zf_image + self.modl_lamda * image
            CG_alg = ConjGrad(Aop_fun=Sense.normal,b=rhs,verbose=False,l2lam=self.modl_lamda,max_iter=self.num_cg_steps)
            image = CG_alg.forward(rhs)
        
        return image
