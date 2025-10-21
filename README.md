# WITM
Fast Pre-Reconstruction for Multimode Fiber Speckle Imaging Using Multi-Wavelength Enhanced Matrix and Bilateral PCA
Overview

This MATLAB code implements a fast pre-reconstruction framework for multimode fiber (MMF) speckle imaging.
The method integrates multi-wavelength information into an enhanced measurement matrix, performs PCA on both speckle (input) and label (output) data, and solves the inverse transmission matrix (ITM) in the reduced subspace for efficient reconstruction.

**Key Features**

**1. Multi-Wavelength Fusion**

Combines speckle patterns captured under different illumination wavelengths (e.g., RGB or 3 spectral bands) to form a richer, more stable input representation.

**2. Bilateral PCA**

Label PCA: Compresses output images while retaining most variance.

Speckle PCA: Reduces high-dimensional input speckle features, improving computational efficiency and robustness.

**3. Inverse Transmission Matrix (ITM)**

Computes a linear mapping from reduced speckle space to label space using pseudoinverse operations.

Enables fast pre-reconstruction without iterative optimization or deep learning.

**4. GPU Acceleration**

PCA decomposition, projections, and reconstruction are GPU-enabled for large datasets.



Save Outputs

Reconstructed images are saved in organized folders as JPEG files.
