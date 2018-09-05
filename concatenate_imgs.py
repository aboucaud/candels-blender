from pathlib import Path

import tqdm
import click
import numpy as np

from blender import segmap

IMG_TMP = 'blend_{:06d}.npy'
SEG_TMP = 'blend_seg_{:06d}.npy'
IMG_DTYPE = np.float32
SEG_DTYPE = np.uint8


def concatenate(path):
    n_img = len(list(path.glob('blend_seg*npy')))

    img0 = np.load(path / IMG_TMP.format(0))
    imgmain = np.empty((n_img, *img0.shape), dtype=img0.dtype)

    for idx in tqdm.trange(n_img):
        imgmain[idx] = np.load(path / IMG_TMP.format(idx))

    np.save(path / 'images.npy', imgmain.astype(IMG_DTYPE))


def concatenate_seg(path, method=None):
    n_img = len(list(path.glob('blend_seg*npy')))

    method = method or segmap_identity

    if isinstance(method, str):
        method = getattr(segmap, method)

    img0 = method(np.load(path / SEG_TMP.format(0)))
    imgmain = np.empty((n_img, *img0.shape), dtype=img0.dtype)

    for idx in tqdm.trange(n_img):
        seg = np.load(path / SEG_TMP.format(idx))
        imgmain[idx] = method(seg)

    np.save(path / 'labels.npy', imgmain.astype(SEG_DTYPE))


def segmap_identity(array):
    return array.astype(SEG_DTYPE)


@click.command()
@click.argument('image_dir')
@click.option('-m', '--method', default=None,
              type=click.Choice(['segmap_encoding_v1', 'segmap_encoding_v2']),
              help="Segmentation method")
def main(image_dir: str, method: str):
    path = Path.cwd() / image_dir
    image_file = path / 'images.npy'
    label_file = path / 'labels.npy'

    if not image_file.exists():
        concatenate(path)
        click.echo('Stamps concatenated !')

    if not label_file.exists():
        concatenate_seg(path, method=method)
        click.echo('Segmentation maps concatenated !')


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
