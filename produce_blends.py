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
@click.option('-d', '--datapath',
              type=click.Path(exists=True),
              help='Path to data files.',
              default='./data', show_default=True)
@click.option('-s', '--seed',
              type=int, help='Random seed.',
              default=42, show_default=True)
def main(n_blend: int, datapath: str, seed: int) -> None:
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
                      magdiff=2, raddiff=4, seed=seed)

    blender.make_cut(blender.cat.mag > 18)
    blender.make_cut(blender.cat.mag < 23)
    blender.make_cut(blender.cat.galtype != 'irr')

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
    main()
