import csv
import logging
from pathlib import Path

import click
import numpy as np

from blender import Blender, Blend
from blender.catalog import blend2cat, CATALOG_HEADER
from blender.segmap import mask_out_pixels

from astropy.visualization import simple_norm
import matplotlib.pyplot as plt


def plot_img(img, seg, gal_idx, idx: int, outdir: str = '.') -> None:
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(18, 6))
    norm = simple_norm(img, stretch='log')
    axes[0].imshow(img)
    axes[1].imshow(img, norm=norm)
    axes[1].set_title(f"gal_id={gal_idx}")
    axes[2].imshow(seg, cmap='gray')
    for ax in axes:
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f'{outdir}/indiv_galaxy_{idx:06d}.png', dpi=300)
    plt.close()


def create_full_set(blender, outdir):
    outdir = outdir / 'check_images'
    if not outdir.exists():
        outdir.mkdir()

    n_imgs = len(blender.data)

    msg = f'Producing visualizations'
    with click.progressbar(range(n_imgs), label=msg) as bar:
        for img_id in bar:
            img = blender.data[img_id].copy()
            seg = blender.seg[img_id].copy()
            segval = seg[64, 64]

            masked_img = mask_out_pixels(img, seg, segval)
            masked_seg = blender.clean_seg(img_id)
            plot_img(masked_img, masked_seg, img_id, img_id, outdir)


def create_image_set(blender: Blender, outdir: Path, test_set=False) -> None:
    """
    Use a Blender instance to output stamps of blended galaxies and
    their associated segmentation mask, plus a catalog of these sources.

    Parameters
    ----------
    blender:
        the Blender instance
    n_blends:
        number of desired images
    outdir:
        output directory
    test_set: default False
        switch between the training and testing galaxy split

    """
    if test_set:
        indices = blender.test_idx
        outdir = outdir / 'test'
    else:
        indices = blender.train_idx
        outdir = outdir / 'train'

    if not outdir.exists():
        outdir.mkdir()

    n_imgs = len(indices)

    msg = f'Producing visualizations'
    with click.progressbar(range(n_imgs), label=msg) as bar:
        for img_id in bar:
            gal_id = indices[img_id]
            gal = blender.galaxy(gal_id)
            img, seg = blender.masked_stamp(gal)
            plot_img(img, seg, gal_id, img_id, outdir)


@click.command()
@click.argument('n_blend', type=int)
@click.option('--mag_low',
              type=float, default=18, show_default=True,
              help='Lowest galaxy magnitude')
@click.option('--mag_high',
              type=float, default=23, show_default=True,
              help='Highest galaxy magnitude')
@click.option('--mag_diff',
              type=float, default=2, show_default=True,
              help='Top magnitude difference between galaxies')
@click.option('--rad_diff',
              type=float, default=4, show_default=True,
              help='Top distance between galaxies as a fraction of radius')
@click.option('-t', '--test_ratio',
              type=float, default=0.2, show_default=True,
              help='Ratio of the input galaxies used only for the test set')
@click.option('-e', '--excluded_type',
              type=click.Choice(['irr', 'disk', 'sph', 'sphd']),
              multiple=True,
              help='Excluded galaxy types')
@click.option('-d', '--datapath',
              type=click.Path(exists=True),
              default='./data', show_default=True,
              help='Path to data files')
@click.option('-s', '--seed',
              type=int, default=42, show_default=True,
              help='Random seed')
def main(n_blend, excluded_type, mag_low, mag_high,
         mag_diff, rad_diff, test_ratio, datapath, seed):
    """
    Produce N_BLEND stamps of HST blended galaxies with their individual masks
    """
    # Define the various paths and create directories
    cwd = Path.cwd()
    datapath = cwd / datapath
    instamps = datapath / 'candels.npy'
    insegmaps = datapath / 'candels_seg.npy'
    incat = datapath / 'candels.csv'

    outdir = cwd / f'output-s_{seed}-n_{n_blend}'
    if not outdir.exists():
        outdir.mkdir()

    blender = Blender(instamps, insegmaps, incat,
                      train_test_ratio=test_ratio,
                      magdiff=mag_diff, raddiff=rad_diff, seed=seed)

    # Apply cuts to the galaxy catalog
    click.echo(
        "Selecting galaxies in the magnitude "
        f"range {mag_low} < m < {mag_high}")
    blender.make_cut(blender.cat.mag > mag_low)
    blender.make_cut(blender.cat.mag < mag_high)
    for galtype in set(excluded_type):
        click.echo(f"Excluding {galtype} galaxies")
        blender.make_cut(blender.cat.galtype != galtype)

    # Compute the train/test splits
    n_test = int(test_ratio * n_blend)
    n_train = n_blend - n_test

    # create_image_set(blender, outdir, test_set=False)
    # create_image_set(blender, outdir, test_set=True)
    create_full_set(blender, outdir)

    click.echo(message=f"Images stored in {outdir}")


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
