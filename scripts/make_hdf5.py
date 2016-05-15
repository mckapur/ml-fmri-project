import numpy as np
import nibabel as nib
import os
import csv
import h5py
import random as rand
import scipy.ndimage as ndimage
import threading

# Constants
STANDARD_TIME_STEP = 3 # 3 seconds is the standard time step between MRI scan images
STANDARD_MRI_DIMENSIONALITY = (61, 73, 61) # The standard dimensionality that each MRI scan should be modelled in and conform to

# Extract metadata from CSV form and format for use
metadata = {}
with open('./../data/abide_metadata.csv', 'rU') as csvfile:
    abide_attributes = csv.reader(csvfile, delimiter=';')
    is_headers = True
    for row in abide_attributes:
        if not is_headers:
            autistic = not (int(row[7]) - 1) # Turn 2/1 to 1/0, then from 1/0 to 0/1
            metadata[row[6]] = {'is_autistic': autistic, 'is_failure': bool(int(row[len(row) - 2]))}
        else:
            is_headers = False


# HDF5 database setup
hdf5_dir = './../data/hdf-5/' # Path to HDF5 folder
hdf5_db = h5py.File(hdf5_dir + 'database.hdf5', 'w') # Create HDF5 file

# Setup HDF5 groups
# Each sample will have its own path, and each matrix being stored will have dimensions 1xXxYxZxT. The corresponding directory structure for the learning sets will be: /{autistic|control}/{train|val|test}/NUM, where NUM is an index for each individual 5-D MRI scan.
aut_hdf5_grp = hdf5_db.create_group('autistic')
aut_train_hdf5_grp = aut_hdf5_grp.create_group('train')
aut_val_hdf5_grp = aut_hdf5_grp.create_group('val')
aut_test_hdf5_grp = aut_hdf5_grp.create_group('test')

con_hdf5_grp = hdf5_db.create_group('control')
con_train_hdf5_grp = con_hdf5_grp.create_group('train')
con_val_hdf5_grp = con_hdf5_grp.create_group('val')
con_test_hdf5_grp = con_hdf5_grp.create_group('test')

# Based on CSV metadata file
aut_m = 448
con_m = 509

# Counts
aut_train_m = 0
aut_val_m = 0
aut_test_m = 0
con_train_m = 0
con_val_m = 0
con_test_m = 0

# Create HDF5 dataset for a specific file
def create_hdf5_dataset(f):
    f_id = f[:-20] # Strip off "_func_minimal.nii.gz" to get file_id
    if metadata[f_id]['is_failure'] == True: # If the MRI quality is poor (a failure), discard
        return
    # Fetch labels and raw data
    mri_file = nib.load(mri_dir + f) # Convert .nii.gz file to readable version
    mri_data = mri_file.get_data() # Load mri from data file
    mri_label = metadata[f_id]['is_autistic'] # Label of MRI as belonging to either an autistic or control person

    mri_data_shape = mri_data.shape
    if not mri_data_shape[0:3] == STANDARD_MRI_DIMENSIONALITY: # If the MRI scan is not of standard dimensionality 61x73x61xT, remove it
        return

    # Normalize time axis to the standardized time step
    mri_time_step = mri_file.header['pixdim'][4]
    if mri_time_step > STANDARD_TIME_STEP: # We can't handle this!
        return
    projected_timesteps = int(mri_data_shape[3]*mri_time_step/STANDARD_TIME_STEP) + 1 # Projected dimensionality/size of T in 1xXxYxZxT
    mri_data_norm = np.zeros(shape=(mri_data_shape[0:3] + (projected_timesteps,))) # Create the normalized MRI scan with the same dimensions for XxYxZ
    mri_data_norm_m = 0
    elapsed_from_standard = STANDARD_TIME_STEP # Add the first timestep
    for i in xrange(mri_data_shape[3]):
        if elapsed_from_standard >= STANDARD_TIME_STEP:
            mri_data_norm_m += 1
            mri_curr = mri_data[:, :, :, i]
            #  Perform interpolation if not at exact timestep
            if elapsed_from_standard > STANDARD_TIME_STEP:
                mri_prev = mri_data[:, :, :, i - 1]
                X, Y, Z = np.meshgrid(np.arange(mri_data_shape[0]), np.arange(mri_data_shape[1]), np.arange(mri_data_shape[2])) # Define grid coordinates where interpolation occurs for X, Y, Z (we use .arange() so it considers every single point)
                discrepency_curr = elapsed_from_standard - STANDARD_TIME_STEP
                discrepency_prev = mri_time_step - discrepency_curr # STANDARD_TIME_STEP - (elapsed_from_standard - mri_time_step) = STANDARD_TIME_STEP - elapsed_from_standard + mri_time_step = mri_time_step - discrepency_curr
                norm_weight = discrepency_prev / (discrepency_prev + discrepency_curr) # norm_weight defines the perentage weight for mri_curr contributing to the mri_norm over mri_prev
                norm_coordinates = np.ones(X.shape) * norm_weight, X, Y, Z
                mri_interpol = np.reshape(ndimage.map_coordinates(np.array([mri_prev, mri_curr]), norm_coordinates), mri_data.shape[0:3])  # Apply interpolation for new mri based on mri_prev and mri_curr
                mri_data_norm[:, :, :, mri_data_norm_m] = mri_interpol
            else: # Effectively using a norm weight of 1, increases efficiency to skip over normalization procedure
                mri_data_norm[:, :, :, mri_data_norm_m] = mri_curr
            elapsed_from_standard -= STANDARD_TIME_STEP
        elapsed_from_standard += mri_time_step
    mri_data = mri_data_norm
    mri_data = np.expand_dims(mri_data, 0) # Reshape to 5-D format of: 1xXxYxZxT

    # Create HDF5 dataset to correct directory
    if mri_label == True:
        if aut_train_m < int(aut_m * 0.8): # 80% of autistic members
            aut_train_m += 1
            aut_train_hdf5_grp.create_dataset(str(aut_train_m), data=mri_data)
        elif aut_val_m < int(aut_m * 0.9): # 10% of autistic members
            aut_val_m += 1
            aut_val_hdf5_grp.create_dataset(str(aut_val_m), data=mri_data)
        elif aut_test_m < int(aut_m): # 10% of autistic members
            aut_test_m += 1
            aut_test_hdf5_grp.create_dataset(str(aut_test_m), data=mri_data)
    else:
        if con_train_m < int(con_m * 0.8): # 80% of control members
            con_train_m += 1
            con_train_hdf5_grp.create_dataset(str(con_train_m), data=mri_data)
        elif con_val_m < int(con_m * 0.9): # 10% of control members
            con_val_m += 1
            con_val_hdf5_grp.create_dataset(str(con_val_m), data=mri_data)
        elif con_test_m < int(con_m): # 10% of control members
            con_test_m += 1
            con_test_hdf5_grp.create_dataset(str(con_test_m), data=mri_data)

# Extract raw data and format it for use
mri_dir = './../data/raw-mris/' # All data in ./../data/raw-mris
for f in os.listdir(mri_dir):
    if not f.endswith('.nii.gz'): # Iterate through all files of format .nii.gz
        continue
    create_hdf5_dataset(f)
        
# Setup metadata matrix dataset
# "A /metadata vector with 6 values (in this order): number of autistic training samples, control training samples, autistic validation samples, control validation samples, autistic test samples, and control test samples."
metadata_hdf5 = np.array([aut_train_m, con_train_m, aut_val_m, con_val_m, aut_test_m, con_test_m])
hdf5_db.create_dataset('metadata', data=metadata_hdf5)
