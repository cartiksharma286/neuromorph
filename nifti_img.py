import numpy as np
# Set numpy to print only 2 decimal digits for neatness
np.set_printoptions(precision=2, suppress=True)

import os
import nibabel as nib
from nibabel.testing import data_path

example_ni1 = os.path.join(data_path, 'brainweb.mnc.gz')
n1_img = nib.load(example_ni1)
n1_img

example_ni2 = os.path.join(data_path,'example_nifti.nii.gz')
nd_img = nib.load(example_ni2)

nib.save(nd_img, 'scaled_image.nii')
