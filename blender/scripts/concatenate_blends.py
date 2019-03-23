import os
from pathlib import Path
from typing import Callable

import click
import numpy as np  # type: ignore

from blender import segmap

IMG_TMP = "{prefix}_blend_{idx:06d}.npy"
SEG_TMP = "{prefix}_blend_seg_{idx:06d}.npy"
IMG_DTYPE = np.float32
SEG_DTYPE = np.uint8


def concatenate_blends(n_img: int, filepath: Path, prefix: str) -> None:
    """
    Create a stack of blends from the individual images.

    The individual files actually contain the images of the individual
    galaxies that need to be sum up to obtain the blend. This method
    takes care of this and can also output the individual galaxies as the
    labels.

    Parameters
    ----------
    path:
        directory containing the individual image files
    prefix: {'train','test'}
        prefix of the image files corresponding to the split
    with_labels:
        saves the individual images as target

    """
    datadir = filepath.parent
    # Retrieving shape of images for the output
    img0 = np.load(datadir / IMG_TMP.format(prefix=prefix, idx=0))

    # Create placeholder for the images
    imgmain = np.empty((n_img, *img0.shape[:-1]), dtype=img0.dtype)

    # Load and process the images
    msg = f"Processing the {prefix}ing blends"
    with click.progressbar(range(n_img), label=msg) as bar:
        for idx in bar:
            img = np.load(datadir / IMG_TMP.format(prefix=prefix, idx=idx))
            imgmain[idx] = img.sum(axis=-1)

    np.save(filepath, imgmain.astype(IMG_DTYPE))


def concatenate_single_images(n_img: int, filepath: Path, prefix: str):
    """
    Create a stack of blends from the individual images.

    The individual files actually contain the images of the individual
    galaxies that need to be sum up to obtain the blend. This method
    takes care of this and can also output the individual galaxies as the
    labels.

    Parameters
    ----------
    path:
        directory containing the individual image files
    prefix: {'train','test'}
        prefix of the image files corresponding to the split
    with_labels:
        saves the individual images as target

    """
    datadir = filepath.parent
    # Retrieving the shape of images for the output
    img0 = np.load(datadir / IMG_TMP.format(prefix=prefix, idx=0))

    # Create placeholder for the images
    img_indiv = np.empty((n_img, *img0.shape), dtype=img0.dtype)

    # Load and process the images
    msg = f"Processing the {prefix}ing single images"
    with click.progressbar(range(n_img), label=msg) as bar:
        for idx in bar:
            # Channels last
            img_indiv[idx, ...] = np.load(datadir / IMG_TMP.format(prefix=prefix, idx=idx))

    np.save(filepath, img_indiv.astype(IMG_DTYPE))


def concatenate_masks(n_img: int, filepath: Path, prefix: str,
                      method: str) -> None:
    """
    Create a stack of masks from the individual files.

    The input files contain the masks of the individual galaxies which are
    then processed using the specified method to create the target labels.

    Parameters
    ----------
    path:
        directory containing the individual segmentation files
    prefix: {'train','test'}
        prefix of the image files corresponding to the split
    method: {'bogg_masks', 'ogg_masks', 'gg_masks}
        name of existing methods in `blender.segmap` to produce the labels

    """
    datadir = filepath.parent

    mask_builder: Callable = getattr(segmap, method)

    # Retrieving the shape of images for the output
    img0 = mask_builder(np.load(datadir / SEG_TMP.format(prefix=prefix, idx=0)))

    # Create placeholder for the images
    imgmain = np.empty((n_img, *img0.shape), dtype=img0.dtype)

    # Load and process the images
    msg = f"Processing the {prefix}ing masks"
    with click.progressbar(range(n_img), label=msg) as bar:
        for idx in bar:
            seg = np.load(datadir / SEG_TMP.format(prefix=prefix, idx=idx))
            imgmain[idx] = mask_builder(seg)

    np.save(filepath, imgmain.astype(SEG_DTYPE))


@click.command("concatenate")
@click.option(
    "-d",
    "--image_dir",
    metavar="<image-dir>",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "-m",
    "--method",
    type=click.Choice([
        "bogg_masks",
        "ogg_masks",
        "gg_masks",
        "single_images"
    ]),
    required=True,
)
@click.option("--delete", is_flag=True, help="Delete individual images once finished")
def main(image_dir, method, delete):
    """
    Concatenate the individual blended sources and masks from <image-dir>
    to create binary files with blends and targets.

    \b
    Four methods are available to obtain various targets, depending on the goal:
    - `gg_masks` for Galaxy, Galaxy masks
    - `ogg_masks` for Overlap, Galaxy, Galaxy masks
    - `bogg_masks` for Background, Overlap, Galaxy, Galaxy masks
    - `single_images` for the individual galaxy stamps.

    Details for the various mask methods can be found in `blender/segmap.py`

    Use the --delete option to remove the individual image files at the end.

    """
    datadir = Path.cwd() / image_dir

    for prefix in ["train", "test"]:
        n_img = len(list(datadir.glob(f"{prefix}_blend_seg_*npy")))

        blend_file = datadir / f"{prefix}_blends.npy"
        target_file = datadir / f"{prefix}_{method}.npy"

        if not blend_file.exists():
            concatenate_blends(n_img, blend_file, prefix)
            click.echo(f"=> {blend_file} created")

        if not target_file.exists():
            if method == "single_images":
                concatenate_single_images(n_img, target_file, prefix)
            else:
                concatenate_masks(n_img, target_file, prefix, method=method)
            click.echo(f"=> {target_file} created")

        if delete:
            for img in datadir.glob(f"{prefix}_blend_*.npy"):
                os.remove(img)
            click.echo(f"Individual {prefix} stamps deleted")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
