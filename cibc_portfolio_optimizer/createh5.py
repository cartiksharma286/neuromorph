import numpy as np
import h5py

# Create a NumPy array
#data_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
data_array = np.load("gt_kspace.npy")

# Define the path to the HDF5 file
hdf5_file_path = 'data.h5'

# Save the NumPy array to an HDF5 file
with h5py.File(hdf5_file_path, 'w') as hdf5_file:
    hdf5_file.create_dataset('dataset', data=data_array)
hdf5_file.close()
