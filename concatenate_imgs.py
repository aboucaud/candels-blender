from pathlib import Path

import tqdm
import click
import numpy as np


IMG_TMP = 'blend_{:06d}.npy'
SEG_TMP = 'blend_seg_{:06d}.npy'
IMG_DTYPE = np.float32
SEG_DTYPE = np.uint8


def one_hot_encode(segmap):
    """Convert label array to one hot encoding as defined in the UNet"""
    segmap = segmap.astype(bool)
    s1, s2 = segmap
    array_list = [~s1 & ~s2,     # background
                  s1 & s2,       # overlap
                  s1 ^ s1 & s2,  # s1 without overlap
                  s2 ^ s1 & s2]  # s2 without overlap

    output = np.concatenate([np.expand_dims(arr, axis=-1)
                             for arr in array_list], axis=2)

    return output


def new_one_hot_encode(segmap):
    segmap = segmap.astype(bool)
    s1, s2 = segmap
    array_list = [
        np.logical_and(s1, s2),
        s1,
        s2
    ]

    output = np.concatenate(
        [np.expand_dims(arr, axis=-1)
         for arr in array_list], 
        axis=2)

    return output.astype(SEG_DTYPE)


def concatenate(path):
    n_img = len(list(path.glob('blend_seg*npy')))

    img0 = np.load(path / IMG_TMP.format(0))
    imgmain = np.empty((n_img, *img0.shape), dtype=img0.dtype)

    for idx in tqdm.trange(n_img):
        imgmain[idx] = np.load(path / IMG_TMP.format(idx))

    np.save(path / 'images.npy', imgmain.astype(IMG_DTYPE))


def concatenate_seg(path, new=False):
    n_img = len(list(path.glob('blend_seg*npy')))

    method = one_hot_encode
    if new:
        method = new_one_hot_encode

    img0 = method(np.load(path / SEG_TMP.format(0)))
    imgmain = np.empty((n_img, *img0.shape), dtype=img0.dtype)

    for idx in tqdm.trange(n_img):
        seg = np.load(path / SEG_TMP.format(idx))
        imgmain[idx] = new_one_hot_encode(seg)

    np.save(path / 'labels.npy', imgmain.astype(SEG_DTYPE))


@click.command()
@click.argument('image_dir')
def main(image_dir: str):
    path = Path.cwd() / image_dir
    image_file = path / 'images.npy'
    label_file = path / 'labels.npy'

    if not image_file.exists():
        concatenate(path)
        click.echo('Stamps concatenated !')

    if not label_file.exists():
        concatenate_seg(path)
        click.echo('Segmentation maps concatenated !')


if __name__ == '__main__':
    main()
