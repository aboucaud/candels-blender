import csv
import pathlib
import logging

import tqdm
import click
import numpy as np

from blender import Blender, Blend
from blender.catalog import blend2cat, CATALOG_HEADER


def save_img(blend: Blend, idx: int, outdir: str = '.') -> None:
    np.save(f'{outdir}/blend_{idx:06d}.npy', blend.img)
    np.save(f'{outdir}/blend_seg_{idx:06d}.npy', blend.segmap)


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
@click.option('-e', '--excluded',
              type=click.Choice(['irr', 'disk', 'sph', 'sphd']),
              multiple=True, default=('irr',), show_default=True,
              help='Excluded galaxy types')
@click.option('-d', '--datapath',
              type=click.Path(exists=True),
              default='./data', show_default=True,
              help='Path to data files')
@click.option('-s', '--seed',
              type=int, default=42, show_default=True,
              help='Random seed')
def main(n_blend, excluded, mag_low, mag_high,
         mag_diff, rad_diff, datapath, seed):
    """
    Script that produces N_BLEND stamps of HST blended galaxies
    """
    n_blend = int(n_blend)

    cwd = pathlib.Path.cwd()

    datapath = cwd / datapath
    instamps = datapath / 'candels.npy'
    insegmaps = datapath / 'candels_seg.npy'
    incat = datapath / 'candels.csv'

    outdir = cwd / f'output-s_{seed}-n_{n_blend}'
    if not outdir.exists():
        outdir.mkdir()
    outcat = outdir / 'blend_cat.csv'
    outlog = outdir / 'blender.log'

    logging.basicConfig(
        filename=outlog,
        level=logging.INFO,
        format='%(asctime)s [ %(levelname)s ] : %(message)s')
    outcat = outdir / 'blend_cat.csv'

    blender = Blender(instamps, insegmaps, incat,
                      magdiff=mag_diff, raddiff=rad_diff, seed=seed)

    click.echo(
        "Selecting galaxies in the magnitude "
        f"range {mag_low} < m < {mag_high}")
    blender.make_cut(blender.cat.mag > mag_low)
    blender.make_cut(blender.cat.mag < mag_high)
    for galtype in set(excluded):
        click.echo(f"Excluding {galtype} galaxies")
        blender.make_cut(blender.cat.galtype != galtype)

    with open(outcat, 'w') as f:
        output = csv.writer(f)
        output.writerow(CATALOG_HEADER)
        for blend_id in tqdm.trange(n_blend):
            blend = blender.next_blend()
            while blend is None:
                blend = blender.next_blend()
            output.writerow(blend2cat(blend, blend_id))
            save_img(blend, blend_id, outdir)

    click.echo(message=f"Images stored in {outdir}")


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
