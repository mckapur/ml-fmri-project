import numpy as np
import nibabel as nib
import os
import csv
import h5py
import random as rand
import scipy.ndimage as ndimage

# Constants
STANDARD_TIME_STEP = 3 # 3 seconds is the standard time step between MRI scan images

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

# Extract raw data and format it for use

autistic_members = [] # Array of MRI scans of austistic label
control_members = [] # Array of MRI scans of control (non-autistic) label

mri_dir = './../data/raw-mris/' # All data in ./../data/raw-mris
for f in os.listdir(mri_dir):
    if not f.endswith('.nii.gz'): # Iterate through all files of format .nii.gz
        continue
    f_id = f[:-20] # Strip off "_func_minimal.nii.gz" to get file_id
    if metadata[f_id]['is_failure'] == True: # If the MRI quality is poor (a failure), discard
        continue

    # Fetch labels and raw data
    mri_file = nib.load(mri_dir + f) # Convert .nii.gz file to readable version
    mri_data = mri_file.get_data() # Load mri from data file
    mri_label = metadata[f_id]['is_autistic'] # Label of MRI as belonging to either an autistic or control person

    mri_data_shape = mri_data.shape
    if not mri_data_shape[0:3] == (61, 73, 61): # If the MRI scan is not of standard dimensionality 61x73x61xT, remove it
        continue

    # Normalize time axis to the standardized time step
    mri_time_step = mri_file.header['pixdim'][4]
    mri_data_norm = np.zeros(shape=(mri_data_shape[0:3] + (0,))) # Create the normalized MRI scan with the same dimensions for XxYxZ
    elapsed_from_standard = STANDARD_TIME_STEP # Add the first timestep
    for i in xrange(mri_data_shape[3]):
        if elapsed_from_standard >= STANDARD_TIME_STEP:
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
                mri_data_norm = np.append(mri_data_norm, mri_interpol)
            else: # Effectively using a norm weight of 1, increases efficiency to skip over normalization procedure
                mri_data_norm = np.append(mri_data_norm, mri_curr)
            elapsed_from_standard = 0
        elapsed_from_standard += mri_time_step
    mri_data = mri_data_norm
    mri_data = np.expand_dims(mri_data, 0) # Reshape to 5-D format of: 1xXxYxZxT

    # Append to correct label-wise container
    if mri_label == True:
        autistic_members.append(mri_data)
    else:
        control_members.append(mri_data)
        
# Shuffle MRI scan arrays for random sampling
rand.shuffle(autistic_members)
rand.shuffle(control_members)

# Setup learning sets (training, test, validation)

aut_m = len(autistic_members) # Number of members in the autistic pool
con_m = len(control_members) # Number of members in the control pool

aut_train_set = autistic_members[:int(aut_m*0.8)] # 80% of autistic_members
aut_val_set = autistic_members[int(aut_m*0.8):int(aut_m*0.9)] # 10% of autistic_members
aut_test_set = autistic_members[int(aut_m*0.9):] # 10% of autistic_members

con_train_set = control_members[:int(con_m*0.8)] # 80% of control members
con_val_set = control_members[int(con_m*0.8):int(con_m*0.9)] # 10% of control members
con_test_set = control_members[int(con_m*0.9):] # 10% of control members

# HDF5 format generation

hdf5_dir = './../data/hdf-5/' # Path to HDF5 folder
hdf5_db = h5py.File(hdf5_dir + 'database.hdf5', 'w') # Create HDF5 file

# Setup metadata matrix dataset
# "A /metadata vector with 6 values (in this order): number of autistic training samples, control training samples, autistic validation samples, control validation samples, autistic test samples, and control test samples."
metadata_hdf5 = np.array([len(aut_train_set), len(con_train_set), len(aut_val_set), len(con_val_set), len(aut_test_set), len(con_test_set)])
hdf5_db.create_dataset('metadata', data=metadata_hdf5)

# Each sample will then have its own path, and each matrix being stored will have dimensions 1xXxYxZxT. The corresponding directory structure for the learning sets will be: /{autistic|control}/{train|val|test}/NUM, where NUM is an index for each individual 5-D MRI scan.

# Setup HDF5 groups
aut_hdf5_grp = hdf5_db.create_group('autistic')
aut_train_hdf5_grp = aut_hdf5_grp.create_group('train')
aut_val_hdf5_grp = aut_hdf5_grp.create_group('val')
aut_test_hdf5_grp = aut_hdf5_grp.create_group('test')

con_hdf5_grp = hdf5_db.create_group('control')
con_train_hdf5_grp = con_hdf5_grp.create_group('train')
con_val_hdf5_grp = con_hdf5_grp.create_group('val')
con_test_hdf5_grp = con_hdf5_grp.create_group('test')

# Initialize HDF5 datasets
for i in xrange(len(aut_train_set)):
    aut_train_hdf5_grp.create_dataset(str(i + 1), data=aut_train_set[i])
for i in xrange(len(aut_val_set)):
    aut_val_hdf5_grp.create_dataset(str(i + 1), data=aut_val_set[i])
for i in xrange(len(aut_test_set)):
    aut_test_hdf5_grp.create_dataset(str(i + 1), data=aut_test_set[i])

for i in xrange(len(con_train_set)):
    con_train_hdf5_grp.create_dataset(str(i + 1), data=con_train_set[i])
for i in xrange(len(con_val_set)):
    con_val_hdf5_grp.create_dataset(str(i + 1), data=con_val_set[i])
for i in xrange(len(con_test_set)):
    con_test_hdf5_grp.create_dataset(str(i + 1), data=con_test_set[i])
