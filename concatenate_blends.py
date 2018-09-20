import os
from pathlib import Path

import click
import numpy as np

from blender import segmap

IMG_TMP = '{prefix}_blend_{idx:06d}.npy'
SEG_TMP = '{prefix}_blend_seg_{idx:06d}.npy'
IMG_DTYPE = np.float32
SEG_DTYPE = np.uint8


def concatenate_img(path, prefix, with_labels=False):
    n_img = len(list(path.glob(f'{prefix}_blend_seg*npy')))

    img0 = np.load(path / IMG_TMP.format(prefix=prefix, idx=0))
    imgmain = np.empty((n_img, *img0.shape[:-1]), dtype=img0.dtype)
    if with_labels:
        img_indiv = np.empty((n_img, *img0.shape), dtype=img0.dtype)

    msg = f'Processing {prefix} stamps'
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


def concatenate_seg(path, prefix, method=None):
    n_img = len(list(path.glob(f'{prefix}_blend_seg*npy')))

    method = method or segmap_identity

    if isinstance(method, str):
        method = getattr(segmap, method)

    img0 = method(np.load(path / SEG_TMP.format(prefix=prefix, idx=0)))
    imgmain = np.empty((n_img, *img0.shape), dtype=img0.dtype)

    msg = f'Loading {prefix} masks'
    with click.progressbar(range(n_img), label=msg) as bar:
        for idx in bar:
            seg = np.load(path / SEG_TMP.format(prefix=prefix, idx=idx))
            imgmain[idx] = method(seg)

    np.save(path / f'{prefix}_labels.npy', imgmain.astype(SEG_DTYPE))


def segmap_identity(array):
    return array.astype(SEG_DTYPE)


@click.command()
@click.argument('image_dir', type=click.Path(exists=True))
@click.argument('method',
                type=click.Choice(['background_overlap_galaxies',
                                   'overlap_galaxies',
                                   'individual_galaxy_images']))
@click.option('--test', is_flag=True,
              help="Test images")
@click.option('--delete', is_flag=True,
              help="Delete individual images once finished")
def main(image_dir, method, test, delete):
    """
    Concatenate the individual blended sources and masks from IMAGE_DIR
    into two files `images.npy` and `labels.npy`.

    `image.npy` (32 bits) contains the stacked blend images

    `labels.npy` (bool) contains the labels produced from the masks
    with the given METHOD

    """
    prefix = 'train'
    if test:
        prefix = 'test'

    path = Path.cwd() / image_dir
    image_file = path / f'{prefix}_images.npy'
    label_file = path / f'{prefix}_labels.npy'

    if not image_file.exists():
        if method == 'individual_galaxy_images':
            concatenate_img(path, prefix, with_labels=True)
        else:
            concatenate_img(path, prefix)
        click.echo('Stamps concatenated')

    if method != 'individual_galaxy_images':
        if not label_file.exists():
            concatenate_seg(path, prefix, method=method)
            click.echo('Segmentation maps concatenated')

    if delete:
        for img in path.glob(f'{prefix}_blend_*.npy'):
            os.remove(img)
        click.echo(f'Individual {prefix} stamps deleted')


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
