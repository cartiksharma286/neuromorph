# Importing standard Qiskit libraries and configuring account
from qiskit import *
#from qiskit import IBMQ
from qiskit.compiler import transpile, assemble
#from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit_aer import Aer


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
#style.use('bmh')
print("here")

num_qubits = 4
# A 8x8 binary image represented as a numpy array
image = np.array([ [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0] ])
# Function for plotting the image using matplotlib
def plot_image(img, title: str):
    plt.title(title)
    plt.xticks(range(img.shape[0]))
    plt.yticks(range(img.shape[1]))
#    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], 
#cmap='viridis')
#    plt.show()

plot_image(image, 'Original Image')

def amplitude_encode(img_data):

    # Calculate the RMS value
    rms = np.sqrt(np.sum(np.sum(img_data**2, axis=1)))

    # Create normalized image
    image_norm = []
    for arr in img_data:
        for ele in arr:
            if rms==0:
                image_norm.append(0)
            else:
                image_norm.append(ele / rms)

    # Return the normalized image as a numpy array
    return np.array(image_norm)

# Get the amplitude ancoded pixel values
# Horizontal: Original image
image_norm_h = amplitude_encode(image)

# Vertical: Transpose of Original image
image_norm_v = amplitude_encode(image.T)

data_qb = 6
anc_qb = 1
total_qb = data_qb + anc_qb

# Initialize the amplitude permutation unitary
D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)
image_size = 256       # Original image-width
image_crop_size = 8   # Width of each part of image for processing

crops=[]
for i in range(image_size//image_crop_size):
    for j in range(image_size//image_crop_size):        
        crops.append(image[i*image_crop_size:(i+1)*image_crop_size,j*image_crop_size:(j+1)*image_crop_size])

edge_crops=[]
x=0
for crop in crops:
    if not((crop==0).all() or (crop==1).all()):
        # Horizontal: Original image
        image_norm_h = amplitude_encode(crop)

        # Vertical: Transpose of Original image
        image_norm_v = amplitude_encode(crop.T)
        # Create the circuit for horizontal scan
        qc_h = QuantumCircuit(total_qb)
        qc_h.initialize(image_norm_h, range(1, total_qb))
        qc_h.h(0)
        qc_h.unitary(D2n_1, range(total_qb))
        qc_h.h(0)

        # Create the circuit for vertical scan
        qc_v = QuantumCircuit(total_qb)
        qc_v.initialize(image_norm_v, range(1, total_qb))
        qc_v.h(0)
        qc_v.unitary(D2n_1, range(total_qb))
        qc_v.h(0)

        # Combine both circuits into a single list
        circ_list = [qc_h, qc_v]

        # Simulating the cirucits
        back = Aer.get_backend('statevector_simulator')
        results = back.run(circ_list).result()
        sv_h = results.get_statevector(qc_h)
        sv_v = results.get_statevector(qc_v)

        threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)

        # Selecting odd states from the raw statevector and
        # reshaping column vector of size 64 to an 8x8 matrix
        edge_scan_h = np.abs(np.array([1 if threshold(sv_h[2*i+1].real) 
else 0 for i in range(2**data_qb)])).reshape(8, 8)
        edge_scan_v = np.abs(np.array([1 if threshold(sv_v[2*i+1].real) 
else 0 for i in range(2**data_qb)])).reshape(8, 8).T
        edge_scan_sim = edge_scan_h | edge_scan_v

        edge_crops.append(edge_scan_sim)

    else:
        edge_crops.append(crop)

    if x%32==0:
        print(x)
    x+=1
tmps=[]
for j in range(32):
    init=edge_crops[32*j]
    for i in range(1,32):
        init=np.concatenate((init, edge_crops[32*j+i]), axis=1)
    tmps.append(init)

actual_edge_image=tmps[0]
for i in range(1,32):
    actual_edge_image=np.concatenate((actual_edge_image, tmps[i]), axis=0)


plt.title('Big Image')
plt.xticks(range(0, actual_edge_image.shape[0]+1, 32))
plt.yticks(range(0, actual_edge_image.shape[1]+1, 32))
plt.imshow(actual_edge_image, extent=[0, actual_edge_image.shape[0], 
actual_edge_image.shape[1], 0], cmap='viridis')
plt.show()

# Create the circuit for horizontal scan
qc_h = QuantumCircuit(total_qb)
qc_h.initialize(image_norm_h, range(1, total_qb))
qc_h.h(0)
qc_h.unitary(D2n_1, range(total_qb))
qc_h.h(0)
#qc_h.draw('mpl', fold=-1)

# Create the circuit for vertical scan
qc_v = QuantumCircuit(total_qb)
qc_v.initialize(image_norm_v, range(1, total_qb))
qc_v.h(0)
qc_v.unitary(D2n_1, range(total_qb))
qc_v.h(0)
#qc_v.draw('mpl', fold=-1)

# Combine both circuits into a single list
circ_list = [qc_h, qc_v]

# Simulating the cirucits
back = Aer.get_backend('statevector_simulator')
results = back.run(circ_list).result()# change from execute
sv_h = results.get_statevector(qc_h)
sv_v = results.get_statevector(qc_v)

from qiskit.visualization import array_to_latex
print('Horizontal scan statevector:')
#print(np.array(sv_h))
#display(array_to_latex(np.array(sv_h)[:30], max_size=30))
print()
print('Vertical scan statevector:')
#display(array_to_latex(np.array(sv_v)[:30], max_size=30))

# Classical postprocessing for plotting the output

# Defining a lambda function for
# thresholding to binary values
threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)

# Selecting odd states from the raw statevector and
# reshaping column vector of size 64 to an 8x8 matrix
edge_scan_h = np.abs(np.array([1 if threshold(sv_h[2*i+1].real) else 0 for 
i in range(2**data_qb)])).reshape(8, 8)
edge_scan_v = np.abs(np.array([1 if threshold(sv_v[2*i+1].real) else 0 for 
i in range(2**data_qb)])).reshape(8, 8).T

# Plotting the Horizontal and vertical scans
plot_image(edge_scan_h, 'Horizontal scan output')
plot_image(edge_scan_v, 'Vertical scan output')

# Combining the horizontal and vertical component of the result
edge_scan_sim = edge_scan_h | edge_scan_v

# Plotting the original and edge-detected images
plot_image(image, 'Original image')
plot_image(edge_scan_sim, 'Edge Detected image')

from PIL import Image
style.use('default')


image_crop_size = 8   # Width of each part of image for processing

# Load the image from filesystem
image_raw = np.array(Image.open('./oas1.gif'))
print('Raw Image info:', image_raw.shape)
print('Raw Image datatype:', image_raw.dtype)
image_size = image_raw.shape[1]

# Convert the RBG component of the image to B&W image, as a numpy (uint8) 
array
image = []
for i in range(image_size):
    image.append([])
    for j in range(image_size):
#        image[i].append(image_raw[i][j][0] / 255)
         image[i].append(image_raw[i][j] / 255)
image = np.array(image)
print('Image shape (numpy array):', image.shape)


# Display the image
plt.title('Big Image')
plt.xticks(range(0, image.shape[0]+1, 32))
plt.yticks(range(0, image.shape[1]+1, 32))
plt.imshow(image, extent=[0, image.shape[0], image.shape[1], 0], 
cmap='viridis')
plt.show()

# Initialize some global variable for number of qubits
data_qb = 6
anc_qb = 1
total_qb = data_qb + anc_qb

# Initialize the amplitude permutation unitary
D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)
image_size = 256       # Original image-width
image_crop_size = 8   # Width of each part of image for processing

crops=[]
for i in range(image_size//image_crop_size):
    for j in range(image_size//image_crop_size):
        
        crops.append(image[i*image_crop_size:(i+1)*image_crop_size,j*image_crop_size:(j+1)*image_crop_size])

edge_crops=[]
x=0
for crop in crops:
    if not((crop==0).all() or (crop==1).all()):
        # Horizontal: Original image
        image_norm_h = amplitude_encode(crop)

        # Vertical: Transpose of Original image
        image_norm_v = amplitude_encode(crop.T)
        # Create the circuit for horizontal scan
        qc_h = QuantumCircuit(total_qb)
        qc_h.initialize(image_norm_h, range(1, total_qb))
        qc_h.h(0)
        qc_h.unitary(D2n_1, range(total_qb))
        qc_h.h(0)

        # Encode the second pixel whose value is (01100100):                            
        value01 = '01100100'
        
        # Add the NOT gate to set the position at 01:                                   
#        qc_h.x(qc_h.num_qubits-1)


        # Create the circuit for vertical scan
        qc_v = QuantumCircuit(total_qb)
        qc_v.initialize(image_norm_v, range(1, total_qb))
        qc_v.h(0)
        qc_v.unitary(D2n_1, range(total_qb))
        qc_v.h(0)

        
        # Combine both circuits into a single list
        circ_list = [qc_h, qc_v]

        # Simulating the cirucits
        back = Aer.get_backend('statevector_simulator')
        results = back.run(circ_list).result()
        sv_h = results.get_statevector(qc_h)
        sv_v = results.get_statevector(qc_v)

        threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)

        # Selecting odd states from the raw statevector and
        # reshaping column vector of size 64 to an 8x8 matrix
        edge_scan_h = np.abs(np.array([1 if threshold(sv_h[2*i+1].real) 
else 0 for i in range(2**data_qb)])).reshape(8, 8)
        edge_scan_v = np.abs(np.array([1 if threshold(sv_v[2*i+1].real) 
else 0 for i in range(2**data_qb)])).reshape(8, 8).T
        edge_scan_sim = edge_scan_h | edge_scan_v

        edge_crops.append(edge_scan_sim)

    else:
        edge_crops.append(crop)

    if x%32==0:
        print(x)
    x+=1

tmps=[]
for j in range(32):
    init=edge_crops[32*j]
    for i in range(1,32):
        init=np.concatenate((init, edge_crops[32*j+i]), axis=1)
    tmps.append(init)

print(init.shape)
print(len(tmps))

actual_edge_image=tmps[0]
for i in range(1,32):
    actual_edge_image=np.concatenate((actual_edge_image, tmps[i]), axis=0)

plt.title('Big Image')
plt.xticks(range(0, actual_edge_image.shape[0]+1, 32))
plt.yticks(range(0, actual_edge_image.shape[1]+1, 32))
plt.imshow(actual_edge_image, extent=[0, actual_edge_image.shape[0], 
actual_edge_image.shape[1], 0], cmap='viridis')
plt.show()
