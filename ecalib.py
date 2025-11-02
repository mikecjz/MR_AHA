import numpy as np
import scipy.io as sio
import sigpy
from sigpy import mri
import h5py

# Use h5py for MATLAB v7.3 files instead of scipy.io.loadmat
with h5py.File('Data/kspace3D.mat', 'r') as f:
    kspace = f['kspace_1'][:].view(np.complex64)
    
sigpy_device = sigpy.Device(0)
    
espirit = mri.app.EspiritCalib(kspace, calib_width=24, thresh=0.02, kernel_width=6, crop=0.95, max_iter=100, device=sigpy_device, output_eigenvalue=False, show_pbar=True)
