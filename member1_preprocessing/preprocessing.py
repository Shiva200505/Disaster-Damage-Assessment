"""
member1_preprocessing/preprocessing.py
────────────────────────────────────────────────────────────────────────────
Member 1 – Core Preprocessing Utilities

Provides:
  extract_patches   – sliding-window crop of large satellite tiles
  align_images      – homography-based alignment of pre/post images
  normalize         – per-image min-max normalisation to [0, 1]
  oversample_patches– duplicate rare (destroyed) patches to balance classes
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── Patch Extraction ──────────────────────────────────────────────────────────

def extract_patches(
    image: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
) -> List[np.ndarray]:
    """
    Slide a window across a large satellite tile and collect square patches.

    Parameters
    ----------
    image      : H×W×C or H×W numpy array
    patch_size : side length of extracted square patches (pixels)
    stride     : step between successive patch positions

    Returns
    -------
    List of H_p×W_p×C (or H_p×W_p) numpy arrays
    """
    H, W = image.shape[:2]
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(patch)
    return patches


def extract_patch_pairs(
    pre: np.ndarray,
    post: np.ndarray,
    mask: Optional[np.ndarray] = None,
    patch_size: int = 256,
    stride: int = 128,
) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Extract aligned (pre, post, mask) patch triples from a tile.

    Parameters
    ----------
    pre, post  : H×W×C satellite images
    mask       : optional H×W binary change mask
    patch_size : patch side length
    stride     : step between patches

    Returns
    -------
    List of (pre_patch, post_patch, mask_patch) tuples.
    mask_patch is None when mask is None.
    """
    H, W = pre.shape[:2]
    triples = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            s = np.s_[y : y + patch_size, x : x + patch_size]
            m = mask[s] if mask is not None else None
            triples.append((pre[s], post[s], m))
    return triples


# ── Image Alignment ──────────────────────────────────────────────────────────

def align_images(
    reference: np.ndarray,
    moving: np.ndarray,
    max_features: int = 1000,
    good_match_ratio: float = 0.75,
) -> np.ndarray:
    """
    Align *moving* to *reference* using ORB feature matching + homography.

    This corrects minor offsets caused by differences in satellite pass timing
    or sensor positioning between the pre- and post-disaster acquisitions.

    Parameters
    ----------
    reference        : H×W×3 reference image (e.g. pre-disaster)
    moving           : H×W×3 image to align (e.g. post-disaster)
    max_features     : max ORB keypoints to detect
    good_match_ratio : Lowe's ratio test threshold

    Returns
    -------
    Aligned version of *moving* (same dtype and shape as input).
    If alignment fails, *moving* is returned unchanged with a warning.
    """
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    mov_gray = cv2.cvtColor(moving,    cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(mov_gray, None)

    h, w = reference.shape[:2]

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return cv2.resize(moving, (w, h))  # insufficient keypoints — fallback to resize

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m for m, n in raw_matches if m.distance < good_match_ratio * n.distance]

    if len(good) < 4:
        return cv2.resize(moving, (w, h))  # not enough inliers

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H_mat, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    fallback = False
    if H_mat is None:
        fallback = True
    else:
        # Prevent extreme distortions (determinant check)
        det = H_mat[0,0] * H_mat[1,1] - H_mat[0,1] * H_mat[1,0]
        if det < 0.2 or det > 5.0:
            fallback = True

    if fallback:
        return cv2.resize(moving, (w, h))

    aligned = cv2.warpPerspective(moving, H_mat, (w, h))
    return aligned


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalise an image array to the [0, 1] float32 range.

    For uint8 images   → divide by 255.
    For float images   → min-max scale so that min=0, max=1.
    """
    img = image.astype(np.float32)
    if image.dtype == np.uint8:
        return img / 255.0
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


# ── Class-imbalance Oversampling ──────────────────────────────────────────────

def oversample_patches(
    patch_triples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    min_damaged_fraction: float = 0.05,
    oversample_factor: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Duplicate patches where the change mask contains a significant
    fraction of damaged pixels. This helps balance the class distribution.

    Parameters
    ----------
    patch_triples         : list of (pre, post, mask) np.ndarray triples
    min_damaged_fraction  : fraction of positive mask pixels to qualify
    oversample_factor     : how many extra copies to append

    Returns
    -------
    Extended list with oversampled damaged patches appended.
    """
    damaged, normal = [], []
    for triple in patch_triples:
        _, _, m = triple
        if m is not None and m.mean() >= min_damaged_fraction:
            damaged.append(triple)
        else:
            normal.append(triple)

    oversampled = damaged * oversample_factor
    return normal + oversampled
