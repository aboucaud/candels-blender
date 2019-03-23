import numpy as np  # type: ignore
from numpy import ndarray as Stamp  # pragma: no cover
from scipy.ndimage import binary_dilation  # type: ignore


def normalize_segmap(segmap: Stamp) -> Stamp:
    """
    Reindexes the various objects in the current segmap
    """
    new_segmap: Stamp = segmap.copy()
    val_list = np.unique(segmap)
    for idx, val in enumerate(val_list):
        new_segmap[new_segmap == val] = idx
    return new_segmap


    """
    Replace central galaxy neighbours with background noise

    Basic recipe to replace the detected sources around the central galaxy
    with either randomly selected pixels from the background, or a random
    realisation of the background noise.

    """
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


def gg_masks(segmap: Stamp, dtype=np.uint8) -> np.array:
    """
    Returns the given array cast in a specific type.
    """
    return segmap.astype(dtype)


def ogg_masks(segmap: Stamp, dtype=np.uint8) -> Stamp:
    """
    Convert galaxy segmaps to a special encoding to predict overlap region.

    OGG stands for Overlap, Galaxy, Galaxy

    The input segmap is of shape (N, N, 2) where NxN is the dimensension
    of the stamps and the third axes corresponds to the two galaxies,
    ordered as central first and companion second.

    The output segmap is of shape (N, N, 3). The third axis is ordered as
      1) mask of overlapping region
      2) mask of central galaxy
      3) mask of companion galaxy

    """
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


def bogg_masks(segmap: Stamp, dtype=np.uint8) -> Stamp:
    """
    Convert galaxy segmaps to one hot encoding as defined in the UNet.

    BOGG stands for Background, Overlap, Galaxy, Galaxy

    The input segmap is of shape (N, N, 2) where NxN is the dimensension
    of the stamps and the third axes corresponds to the two galaxies,
    ordered as central first and companion second.

    The output segmap is of shape (N, N, 4). The third axis is ordered as
      1) background mask
      2) mask of overlapping region
      3) mask of central galaxy - 2)
      4) mask of companion galaxy - 2)

    This way only each pixel of the NxN blend is assigned one category only.

    """
    segmap = segmap.astype(bool)
    s1, s2 = segmap
    array_list = [~s1 & ~s2,     # background
                  s1 & s2,       # overlap
                  s1 ^ s1 & s2,  # s1 without overlap
                  s2 ^ s1 & s2]  # s2 without overlap

    output = np.concatenate([np.expand_dims(arr, axis=-1)
                             for arr in array_list], axis=-1)

    return output.astype(dtype)
