"""
This module includes functions for computing image quality metrics:
Normalized Root Mean Square Error (NRMSE) and Structural Similarity Index (SSIM).

Efrat Shimron (UC Berkeley, 2021).
"""

import numpy as np
from SSIM_PIL import compare_ssim
from PIL import Image


class error_metrics:
    def __init__(self,I_true,I_pred):
        # convert images from complex to magnitude (we do not want complex data for error calculation)
        self.I_true = np.abs(I_true)  
        self.I_pred = np.abs(I_pred)   
        
    def calc_NRMSE(self):    
        # Reshape the images into vectors
        I_true = np.reshape(self.I_true,(1,-1))   
        I_pred = np.reshape(self.I_pred,(1,-1))               
        # Mean Square Error
        self.MSE = np.square(np.subtract(I_true,I_pred)).mean()       
        # Root Mean Square Error
        self.RMSE = np.sqrt(self.MSE)
        # Normalized Root Mean Square Error
        rr = np.max(I_true) - np.min(I_true) # range
        self.NRMSE = self.RMSE/rr
        
    def calc_SSIM(self):
        # Note: in order to use the function compare_ssim, the images must be converted to PIL format

        # convert the images from float32 to uint8 format
        im1_mag_uint8 = (self.I_true * 255 / np.max(self.I_true)).astype('uint8')
        im2_mag_uint8 = (self.I_pred * 255 / np.max(self.I_pred)).astype('uint8')
        # convert from numpy array to PIL format
        im1_PIL = Image.fromarray(im1_mag_uint8)
        im2_PIL = Image.fromarray(im2_mag_uint8)

        self.SSIM = compare_ssim(im1_PIL, im2_PIL)
