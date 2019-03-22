import csv
import logging
from pathlib import Path

import click
import numpy as np

from blender import Blender, Blend
from blender.catalog import blend2cat, CATALOG_HEADER


def save_img(blend: Blend, idx: int, prefix: str,
             outdir: str = '.') -> None:
    np.save(f'{outdir}/{prefix}_blend_{idx:06d}.npy', blend.img)
    np.save(f'{outdir}/{prefix}_blend_seg_{idx:06d}.npy', blend.segmap)


def create_image_set(blender: Blender, n_blends: int, outdir: Path,
                     test_set: bool = False) -> None:
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
    prefix = 'test' if test_set else 'train'

    outcat = outdir / f'{prefix}_blend_cat.csv'

    with open(outcat, 'w') as f:
        output = csv.writer(f)
        output.writerow(CATALOG_HEADER)

        msg = f'Producing {prefix} blended images'
        with click.progressbar(range(n_blends), label=msg) as bar:
            for blend_id in bar:
                blend = blender.next_blend(from_test=test_set)
                while blend is None:
                    blend = blender.next_blend(from_test=test_set)
                output.writerow(blend2cat(blend, blend_id))
                save_img(blend, blend_id, prefix, outdir)


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
    outlog = outdir / 'blender.log'

    logging.basicConfig(
        filename=outlog,
        level=logging.INFO,
        format='%(asctime)s [ %(levelname)s ] : %(message)s')

    blender = Blender(instamps, insegmaps, incat,
                      train_test_ratio=test_ratio,
                      magdiff=mag_diff, raddiff=rad_diff, seed=seed)

    logger = logging.getLogger(__name__)
    logger.info(
        "\n"
        "Configuration\n"
        "=============\n"
        f"Number of blends: {n_blend}\n"
        f"Seed: {seed}\n"
        "\n"
        "Catalog cuts\n"
        "------------\n"
        f"Excluded galaxy types: {excluded_type}\n"
        f"Lowest magnitude: {mag_low}\n"
        f"Highest magnitude: {mag_high}\n"
        "\n"
        "Blend properties\n"
        "----------------\n"
        f"Top difference in magnitude between galaxies: {mag_diff}\n"
        f"Top distance between galaxies as a fraction of radius: {rad_diff}\n"
        )

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

    create_image_set(blender, n_train, outdir)
    create_image_set(blender, n_test, outdir, test_set=True)

    click.echo(message=f"Images stored in {outdir}")


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
