import numpy as np
from numpy import ndarray as Stamp  # pragma: no cover
from scipy.ndimage import binary_dilation


def normalize_segmap(segmap: Stamp) -> Stamp:
    new_segmap = segmap.copy()
    val_list = np.unique(segmap)
    for idx, val in enumerate(val_list):
        new_segmap[new_segmap == val] = idx
    return new_segmap


def mask_out_pixels(img: Stamp, seg: Stamp, segval: Stamp,
                    noise_factor: int = 1, n_iter: int = 5) -> Stamp:
    bseg = binary_dilation(seg, iterations=n_iter)
    centralseg = binary_dilation(np.where(seg == segval, 1, 0),
                                 iterations=n_iter)
    final_mask = np.logical_xor(bseg, centralseg)
    masked_std = np.std(img * np.logical_not(bseg))
    masked_img = img * ~final_mask
    mask_fill = final_mask * np.random.normal(scale=masked_std, size=img.shape)
    noise_map = np.random.normal(scale=masked_std, size=img.shape)
    new_img = masked_img + mask_fill + noise_factor * noise_map

    return new_img.astype(img.dtype)


def mask_out_pixels_2(img: Stamp, segmap: Stamp, segval: Stamp,
                      n_iter: int = 5) -> Stamp:
    # Create binary masks of all segmented sources
    sources = binary_dilation(segmap, iterations=n_iter)
    noise = np.logical_not(sources)
    # Create binary mask of the central galaxy
    central_source = binary_dilation(np.where(segmap == segval, 1, 0),
                                     iterations=n_iter)
    # Compute the binary mask of all sources BUT the central galaxy
    sources_except_central = np.logical_xor(sources, central_source)
    # Select random pixels from the noise in the image
    n_pixels_to_fill_in = sources_except_central.sum()
    noise_pixels = np.random.choice(img[noise], size=n_pixels_to_fill_in)
    # Fill in the voids with these pixels
    masked_img = img.copy()
    masked_img[sources_except_central] = noise_pixels

    return masked_img


def background_overlap_galaxies(segmap: Stamp, dtype=np.uint8) -> Stamp:
    """Convert label array to one hot encoding as defined in the UNet"""
    segmap = segmap.astype(bool)
    s1, s2 = segmap
    array_list = [~s1 & ~s2,     # background
                  s1 & s2,       # overlap
                  s1 ^ s1 & s2,  # s1 without overlap
                  s2 ^ s1 & s2]  # s2 without overlap

    output = np.concatenate([np.expand_dims(arr, axis=-1)
                             for arr in array_list], axis=-1)

    return output.astype(dtype)


def overlap_galaxies(segmap: Stamp, dtype=np.uint8) -> Stamp:
    segmap = segmap.astype(bool)
    s1, s2 = segmap
    array_list = [
        np.logical_and(s1, s2),  # overlap
        s1,                      # galaxy 1
        s2]                      # galaxy 2

    output = np.concatenate(
        [np.expand_dims(arr, axis=-1)
         for arr in array_list], axis=-1)

    return output.astype(dtype)
