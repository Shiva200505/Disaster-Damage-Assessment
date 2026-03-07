"""
member1_preprocessing/spectral_indices.py
────────────────────────────────────────────────────────────────────────────
Member 1 – Spectral Indices Computation

Computes NDVI (vegetation) and NDWI (water) spectral indices from RGB
satellite imagery. For true multispectral images, the NIR band is band 4;
for standard RGB images we approximate NIR with the red channel when
a dedicated NIR channel is absent.

Functions
─────────
compute_ndvi(red, nir)       → ndvi array in [-1, 1]
compute_ndwi(green, nir)     → ndwi array in [-1, 1]
add_spectral_channels(image) → H×W×5 array (RGB + NDVI + NDWI)
"""

import numpy as np


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red + ε)

    Parameters
    ----------
    red : H×W float32 array, values in [0, 1]
    nir : H×W float32 array, values in [0, 1]

    Returns
    -------
    ndvi : H×W float32 array, values in [-1, 1]
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return np.clip(ndvi, -1.0, 1.0)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Water Index.

    NDWI = (Green - NIR) / (Green + NIR + ε)

    Parameters
    ----------
    green : H×W float32 array, values in [0, 1]
    nir   : H×W float32 array, values in [0, 1]

    Returns
    -------
    ndwi : H×W float32 array, values in [-1, 1]
    """
    green = green.astype(np.float32)
    nir   = nir.astype(np.float32)
    ndwi  = (green - nir) / (green + nir + 1e-8)
    return np.clip(ndwi, -1.0, 1.0)


def add_spectral_channels(image: np.ndarray) -> np.ndarray:
    """
    Append NDVI and NDWI as extra channels to an RGB image.

    For a standard 3-channel RGB image we approximate:
        Red  → channel 0
        Green→ channel 1
        Blue → channel 2
        NIR  → approximated as Red channel (channel 0) when NIR unavailable.

    For a 4-channel (R,G,B,NIR) image, the true NIR band (channel 3) is used.

    Parameters
    ----------
    image : H×W×C numpy array (uint8 or float32)
             C=3 (RGB) or C=4 (RGBN)

    Returns
    -------
    H×W×(C+2) float32 array with NDVI and NDWI appended as channels C and C+1.
    """
    img = image.astype(np.float32)

    # Normalise to [0, 1] if uint8
    if img.max() > 1.0:
        img = img / 255.0

    if img.ndim == 2:
        # Grayscale — stack to 3 channels first
        img = np.stack([img, img, img], axis=-1)

    C = img.shape[-1]

    red   = img[:, :, 0]
    green = img[:, :, 1]
    nir   = img[:, :, 3] if C >= 4 else img[:, :, 0]  # NIR or approximate

    ndvi = compute_ndvi(red=red, nir=nir)    # H×W
    ndwi = compute_ndwi(green=green, nir=nir) # H×W

    # Stack: original channels + NDVI + NDWI → H×W×(C+2)
    result = np.concatenate(
        [img, ndvi[:, :, np.newaxis], ndwi[:, :, np.newaxis]],
        axis=-1,
    )
    return result.astype(np.float32)
