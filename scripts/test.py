import nibabel as nib
import numpy as np
import png
import os
import sys

# def norm(arr, maxval=2**16-1):
#     return arr * maxval / np.max(arr)
#
# f = nib.load('data/CMU_a_0050647_func_minimal.nii.gz')
# data = f.get_data()
#
# for t in range(data.shape[3]):
#     tdata = data[:,:,:,t]
#     os.makedirs(str(t) + 't')
#     for z in range(tdata.shape[2]):
#         png.from_array(norm(tdata[:,:,z]), 'L;16').save(str(t) + 't/' + str(z) + '.png')

for f in os.listdir('data/'):
    if not f.endswith('.nii.gz'):
        continue

    mri = nib.load('data/' + f)
    # Preprocess and load mri into data file
    print mri.get_data().nbytes
