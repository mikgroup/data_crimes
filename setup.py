# setup.py
from setuptools import setup

setup(
    name='subtle_data_crimes',
    version='1.0.0',
    packages=['subtle_data_crimes'],
    install_requires=[
        "data==0.4",
        "h5py==2.10.0",
        "matplotlib==3.3.1",
        "mkl==2021.4.0",
        "mkl-service==2.3.0",
        "numpy==1.19.1",
        "Pillow==8.4.0",
        "progressbar33==2.4",
        "sigpy==0.1.20",
        "SSIM_PIL==1.0.13",
        "torch==1.6.0",  # Get corresponding cuda version: https://pytorch.org.
        "tqdm==4.48.2",
    ],
    dependency_links=['https://github.com/fbcotter/pytorch_wavelets']
)
