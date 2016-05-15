import numpy as np
import nibabel as nib
import os
import csv
import h5py
import random as rand

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

    np.reshape(mri_data, (1, mri_data.shape[0], mri_data.shape[1], mri_data.shape[2], mri_data.shape[3])) # Reshape to 5-D format of: 1xXxYxZxT
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
# Contains: A /metadata vector with 6 values (in this order): number of autistic training samples, control training samples, autistic validation samples, control validation samples, autistic test samples, and control test samples. Each sample will then have its own path, and each matrix being stored will have dimensions 1xXxYxZxT. The corresponding directory structure for the learning sets will be: /{autistic|control}/{train|val|test}/NUM, where NUM is an index for each individual 5-D MRI scan.

hdf5_dir = './../data/hdf-5/' # Path to HDF5 folder
hdf5_db = h5py.File(hdf5_dir + 'database.hdf5', 'w') # Create HDF5 file

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

# Setup metadata matrix dataset
metadata_hdf5 = np.array([len(aut_train_set), len(con_train_set), len(aut_val_set), len(con_val_set), len(aut_test_set), len(con_test_set)])
hdf5_db.create_dataset('metadata', data=metadata_hdf5)
