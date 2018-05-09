import numpy as np

from scipy.ndimage import binary_dilation

__all__ = [
    'normalize_segmap',
    'mask_out_pixels',
    'mask_out_pixels_2',
]

def normalize_segmap(segmap):
    new_segmap = segmap.copy()
    val_list = np.unique(segmap)
    for idx, val in enumerate(val_list):
        new_segmap[new_segmap == val] = idx
    return new_segmap


def mask_out_pixels(img, seg, segval, noise_factor: int = 1, n_iter: int = 5):
    bseg = binary_dilation(seg, iterations=n_iter)
    centralseg = binary_dilation(np.where(seg == segval, 1, 0), iterations=n_iter)
    final_mask = np.logical_xor(bseg, centralseg)
    masked_std = np.std(img * np.logical_not(bseg))
    masked_img = img * ~final_mask
    mask_fill = final_mask * np.random.normal(scale=masked_std, size=img.shape)
    noise_map = np.random.normal(scale=masked_std, size=img.shape)
    new_img = masked_img + mask_fill + noise_factor * noise_map

    return new_img.astype(img.dtype)


def mask_out_pixels_2(img, segmap, segval, n_iter: int = 5):
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
