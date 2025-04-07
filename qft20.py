import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector

def reconstruct_image_from_kspace_classical(k_space_data):
    """Reconstructs an image from k-space using classical IFFT."""
    image_complex = np.fft.ifft2(k_space_data)
    return np.abs(image_complex)

def encode_image_to_amplitudes(image_1d):
    """Encodes a 1D image to quantum amplitudes (power of 2 size)."""
    n_pixels = len(image_1d)
    n_qubits = int(np.ceil(np.log2(n_pixels)))
    if 2**n_qubits != n_pixels:
        raise ValueError("Image size must be a power of 2 for direct amplitude encoding.")
    normalized_image = np.array(image_1d) / np.max(image_1d) if np.max(image_1d) != 0 else np.array(image_1d)
    amplitudes = np.sqrt(normalized_image)
    norm = np.linalg.norm(amplitudes)
    if norm > 1e-9:
        amplitudes /= norm
    return amplitudes, n_qubits

def qft_circuit(num_qubits, inverse=False):
    """Creates a QFT or inverse QFT circuit."""
    qc = QuantumCircuit(num_qubits, name=f"qft{'inv' if inverse else ''}")
    for j in reversed(range(num_qubits)):
        qc.h(j)
        for k in reversed(range(j)):
            phase = -np.pi / (2**(j - k)) if inverse else np.pi / (2**(j - k))
            qc.cp(phase, j, k)
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - 1 - i)
    return qc

def apply_qft_to_image(image_1d):
    """Applies QFT to a 1D image encoded in quantum amplitudes."""
    amplitudes, n_qubits = encode_image_to_amplitudes(image_1d)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(amplitudes, qc.qubits)
    qft = qft_circuit(n_qubits)
    qc.append(qft, qc.qubits)
    simulator = Aer.get_backend('statevector_simulator')
    statevector = Statevector.from_instruction(qc).data
    return np.abs(statevector)**2 # Probabilities (magnitude squared of amplitudes)

def apply_iqft_to_kspace(k_space_quantum):
    """Applies inverse QFT to a quantum k-space representation."""
    n_qubits = int(np.log2(len(k_space_quantum)))
    if 2**n_qubits != len(k_space_quantum):
        raise ValueError("Quantum k-space size must be a power of 2.")
    amplitudes = np.sqrt(k_space_quantum)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(amplitudes, qc.qubits)
    iqft = qft_circuit(n_qubits, inverse=True)
    qc.append(iqft, qc.qubits)
    simulator = Aer.get_backend('statevector_simulator')
    statevector = Statevector.from_instruction(qc).data
    return np.abs(statevector)**2 # Probabilities (magnitude squared of amplitudes)

if __name__ == "__main__":
    # 1. Create a simple 1D "image" (must be power of 2 for direct QFT)
    original_image_1d = np.array([0.1, 0.9, 0.2, 0.8])
    n_pixels = len(original_image_1d)

    # 2. Classical FFT to get k-space
    k_space_classical_1d = np.fft.fft(original_image_1d)

    # 3. Classical IFFT to reconstruct from k-space
    reconstructed_classical_1d = np.fft.ifft(k_space_classical_1d)

    # 4. Quantum FFT to get "quantum k-space"
    try:
        k_space_quantum_1d = apply_qft_to_image(original_image_1d)

        # 5. Quantum Inverse QFT to reconstruct from "quantum k-space"
        reconstructed_quantum_1d = apply_iqft_to_kspace(k_space_quantum_1d)

        # Visualization
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(np.abs(reconstructed_classical_1d), marker='o', label='Classical IFFT')
        plt.plot(original_image_1d, marker='x', linestyle='--', label='Original Image')
        plt.title("Classical Reconstruction")
        plt.xlabel("Pixel Index")
        plt.ylabel("Intensity")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(reconstructed_quantum_1d, marker='o', label='Quantum IQFT')
        plt.plot(original_image_1d, marker='x', linestyle='--', label='Original Image')
        plt.title("Quantum Reconstruction")
        plt.xlabel("Pixel Index")
        plt.ylabel("Probability")
        plt.legend()

        plt.tight_layout()
        plt.show()

        print("Original 1D Image:", original_image_1d)
        print("Classical K-space:", k_space_classical_1d)
        print("Reconstructed Classical Image (Magnitude):", np.abs(reconstructed_classical_1d))
        print("Quantum K-space (Probabilities):", k_space_quantum_1d)
        print("Reconstructed Quantum Image (Probabilities):", reconstructed_quantum_1d)

    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- 2D Example (Classical Only) ---")
    # 6. Create a simple 2D "k-space image"
    size = 8
    k_space_2d = np.zeros((size, size), dtype=complex)
    k_space_2d[size//2 - 1 : size//2 + 1, size//2 - 1 : size//2 + 1] = 1 + 1j

    # 7. Classical IFFT to reconstruct 2D image
    reconstructed_classical_2d = reconstruct_image_from_kspace_classical(k_space_2d)

    # 5. Quantum Inverse QFT to reconstruct from "quantum k-space"                            
#    reconstructed_quantum_1d = apply_iqft_to_kspace(k_space_2d)


    print(reconstructed_quantum_1d)
    # Visualization for 2D
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(k_space_2d), cmap='gray')
    plt.title("2D K-Space (Magnitude)")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_classical_2d, cmap='gray')
    plt.title("Reconstructed 2D Image (Classical)")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
#    print("Reconstructed Classical Image (Magnitude):", np.abs(reconstructed_classical_1d))
#    print("Quantum K-space (Probabilities):", k_space_quantum_1d)
#    print("Reconstructed Quantum Image (Probabilities):", reconstructed_quantum_1d)
    

#    plt.subplot(1, 2, 1)
#    plt.imshow(np.abs(k_space_quantum_1d), cmap='gray')
#    plt.title("2D K-Space (Magnitude)")

#    plt.subplot(1, 2, 2)
#    plt.imshow(reconstructed_quantum_1d, cmap='gray')
#    plt.title("Reconstructed 2D Image (Classical)")

#    plt.tight_layout()
#    plt.show()


    
    print("2D K-space (Magnitude):\n", np.abs(k_space_quantum_1d))
    print("Reconstructed 2D Image (Quantum):\n", reconstructed_quantum_1d)


    def quantum_probabilities_to_image(probabilities):
        """
    Converts probabilities from a quantum statevector (representing an image)
    back to an image-like array.
    
    Args:
        probabilities (np.ndarray): A 1D numpy array of probabilities
                                     obtained from measuring or simulating a
                                     quantum state representing an image.
                                     The length should be a power of 2.

    Returns:
        np.ndarray: A 1D numpy array representing the reconstructed image
                      intensities (scaled to 0-1).
    """
        n_pixels = len(probabilities)
        n_qubits = int(np.log2(n_pixels))
        if 2**n_qubits != n_pixels:
            raise ValueError("Number of probabilities must be a power of 2.")
        
        # The probabilities are proportional to the squared amplitudes.
        # We can take the square root to get amplitudes (relative scale).
        amplitudes = np.sqrt(probabilities)
        
        # Scale the amplitudes to the range 0-1 for image-like intensities.
        # We can normalize them by the maximum amplitude.
        max_amplitude = np.max(amplitudes)
        if max_amplitude > 1e-9:
            image = amplitudes / max_amplitude
        else:
            image = np.zeros_like(amplitudes)
        
        return image

    reconstructed_quantum_image = quantum_probabilities_to_image(reconstructed_quantum_1d)
    print("recon quantum image")
    print(reconstructed_quantum_image)

    print("recon classical image")
    print(reconstructed_classical_2d)
    print("here L282")
    #    plt.imshow(reconstructed_image, cmap='gray')

    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(reconstructed_classical_2d), cmap='gray')                                          
    plt.title("2D K-Space (Magnitude)")

    plt.subplot(1, 2, 2)                                                                         
    plt.imshow(reconstructed_quantum_image, cmap='gray')                                            
    plt.tight_layout()
    plt.show()
