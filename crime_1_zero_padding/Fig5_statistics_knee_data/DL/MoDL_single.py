"""
MoDL for single channel MRI 

(c) Ke Wang (kewang@berkeley.edu), 2020.

"""

import os, sys
import torch
from torch import nn
import sigpy.plot as pl
import utils.complex_utils as cplx
from utils.transforms import SenseModel,SenseModel_single
from utils.layers3D import ResNet
from unet.unet_model import UNet
from utils.flare_utils import ConjGrad
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.use('TkAgg')

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

        # Declare step sizes for each iteration
#         init_step_size = torch.tensor([-2.0], dtype=torch.float32).to(params.device)
#         if fix_step_size:
#             self.step_sizes = [init_step_size] * num_grad_steps
#         else:
#             self.step_sizes = [torch.nn.Parameter(init_step_size) for i in range(num_grad_steps)] 


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
#         CG_alg = ConjGrad(Aop_fun=Sense.normal,b=zf_image,verbose=False,l2lam=0.05,max_iter=self.c)
#         cg_image = CG_alg.forward(zf_image)
#         pl.ImagePlot(zf_image.detach().cpu())

        zf_im_4plot = cplx.to_numpy(zf_image[0,:,:,:].detach().cpu())
        zf_im_4plot = np.rot90(np.abs(zf_im_4plot), 2)
        self.zf_im_4plot = zf_im_4plot

        # # # # debugging plot
        # if self.display_zf_image_flag == 1:
        #     zf_im_4plot = cplx.to_numpy(zf_image[0,:,:,:].detach().cpu())
        #     zf_im_4plot = np.rot90(np.abs(zf_im_4plot), 2)
        #     fig = plt.figure()
        #     plt.imshow(zf_im_4plot,cmap="gray")
        #     plt.title('zf image - check the scale!!!')
        #     plt.colorbar()
        #     plt.show()
        #     filename = self.zf_im_foldername + "/zf_image.png"
        #     fig.savefig(filename)
        #
        #     print('###### max(zf_im)=',np.max(zf_im_4plot),' ######')
        #
        #     print('zf_im saved')
        
#         sys.exit()
        image = zf_image 
        
        # Begin unrolled proximal gradient descent
        for resnet in self.resnets:
            # dc update
            image = image.permute(0,3,1,2) 
#             pl.ImagePlot(image.detach().cpu())
            image = resnet(image)
            image = image.permute(0,2,3,1)
            rhs = zf_image + self.modl_lamda * image
            CG_alg = ConjGrad(Aop_fun=Sense.normal,b=rhs,verbose=False,l2lam=self.modl_lamda,max_iter=self.num_cg_steps)
            image = CG_alg.forward(rhs)
        
        return image
