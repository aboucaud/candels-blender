import os
from pathlib import Path

import click
import numpy as np

from blender import segmap

IMG_TMP = '{prefix}_blend_{idx:06d}.npy'
SEG_TMP = '{prefix}_blend_seg_{idx:06d}.npy'
IMG_DTYPE = np.float32
SEG_DTYPE = np.uint8


def concatenate_img(path: str, prefix: str, with_labels: bool = False):
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
    # Retrieving the number of files and shape of images for the output
    n_img = len(list(path.glob(f'{prefix}_blend_seg*npy')))
    img0 = np.load(path / IMG_TMP.format(prefix=prefix, idx=0))

    # Create placeholder for the images
    imgmain = np.empty((n_img, *img0.shape[:-1]), dtype=img0.dtype)
    if with_labels:
        img_indiv = np.empty((n_img, *img0.shape), dtype=img0.dtype)

    # Load and process the images
    msg = f'Processing the {prefix}ing stamps'
    with click.progressbar(range(n_img), label=msg) as bar:
        for idx in bar:
            img = np.load(path / IMG_TMP.format(prefix=prefix, idx=idx))
            imgmain[idx] = img.sum(axis=-1)
            if with_labels:
                # Channels last
                img_indiv[idx, ...] = img

    np.save(path / f'{prefix}_images.npy', imgmain.astype(IMG_DTYPE))

    if with_labels:
        np.save(path / f'{prefix}_labels.npy', img_indiv.astype(IMG_DTYPE))


def concatenate_seg(path: str, prefix: str, method: str = None):
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
    method: {'background_overlap_galaxies', 'overlap_galaxies'}
        name of existing methods in `blender.segmap` to produce the labels

    """
    method = method or segmap_identity

    if isinstance(method, str):
        method = getattr(segmap, method)

    # Retrieving the number of files and shape of images for the output
    n_img = len(list(path.glob(f'{prefix}_blend_seg*npy')))
    img0 = method(np.load(path / SEG_TMP.format(prefix=prefix, idx=0)))

    # Create placeholder for the images
    imgmain = np.empty((n_img, *img0.shape), dtype=img0.dtype)

    # Load and process the images
    msg = f'Processing the {prefix}ing masks'
    with click.progressbar(range(n_img), label=msg) as bar:
        for idx in bar:
            seg = np.load(path / SEG_TMP.format(prefix=prefix, idx=idx))
            imgmain[idx] = method(seg)

    np.save(path / f'{prefix}_labels.npy', imgmain.astype(SEG_DTYPE))


def segmap_identity(array):
    "Methods that returns the given array in a specific type"
    return array.astype(SEG_DTYPE)


@click.command()
@click.argument('image_dir', type=click.Path(exists=True))
@click.argument('method',
                type=click.Choice(['background_overlap_galaxies',
                                   'overlap_galaxies',
                                   'individual_galaxy_images']))
@click.option('--train', 'prefix', flag_value='train', default=True,
              help="Apply to train images")
@click.option('--test', 'prefix', flag_value='test',
              help="Apply to test images")
@click.option('--delete', is_flag=True,
              help="Delete individual images once finished")
def main(image_dir, method, prefix, delete):
    """
    Concatenate the individual blended sources and masks from IMAGE_DIR
    to create the input (images.npy) and target (labels.npy) files.

    Use the --train/--test option to specify which split to act on.

    Use the --delete option to remove the individual image files at the end.

    """
    path = Path.cwd() / image_dir
    image_file = path / f'{prefix}_images.npy'
    label_file = path / f'{prefix}_labels.npy'

    if not image_file.exists():
        if method == 'individual_galaxy_images':
            concatenate_img(path, prefix, with_labels=True)
        else:
            concatenate_img(path, prefix)
        click.echo(f'=> {image_file} created')

    if method != 'individual_galaxy_images':
        if not label_file.exists():
            concatenate_seg(path, prefix, method=method)
            click.echo(f'=> {label_file} created')

    if delete:
        for img in path.glob(f'{prefix}_blend_*.npy'):
            os.remove(img)
        click.echo(f'Individual {prefix} stamps deleted')


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
