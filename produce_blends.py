import csv
import pathlib
import logging

import tqdm
import numpy as np

from blender import Blender, Blend, Galaxy
from blender.catalog import blend2cat, CATALOG_HEADER


def save_img(blend: Blend, idx: int, outdir: str = '.') -> None:
    np.save(f'{outdir}/blend_{idx:06d}.npy', blend.img)
    np.save(f'{outdir}/blend_seg_{idx:06d}.npy', blend.segmap)


def main(n_blend: int, datapath: str = 'data', seed: int = 42) -> None:
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


if __name__ == '__main__':
    import sys
    main(n_blend=int(sys.argv[1]), seed=int(sys.argv[2]))
