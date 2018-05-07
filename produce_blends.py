import csv
import pathlib
from typing import List

import tqdm
import numpy as np

from blender import Blender, Blend, Galaxy

CAT_HEADER = (
    "id",
    "distance",
    "g1_id",
    "g1_rad",
    "g1_mag",
    "g1_z",
    "g1_type",
    "g2_id",
    "g2_rad",
    "g2_mag",
    "g2_z",
    "g2_type")


def save_img(blend: Blend, idx: int, outdir: str = '.') -> None:
    np.save(f'{outdir}/blend_{idx:06d}.npy', blend.img)
    np.save(f'{outdir}/blend_seg_{idx:06d}.npy', blend.segmap)


def gal2cat(gal: Galaxy) -> List[str]:
    """
    id rad mag z type
    """
    galinfo = [
        f"{gal.gal_id}",
        f"{gal.mag:.6f}",
        f"{gal.rad:.6f}",
        f"{gal.z:.6f}",
        f"{gal.type}"]

    return galinfo


def blend2cat(blend: Blend, idx: int) -> List[str]:
    """
    blend_id distance gal1_id gal1_rad gal1_mag gal1_z gal1_type gal2_id gal2_rad gal2_mag gal2_z gal2_type  # noqa
    """
    distance = np.hypot(*blend.shift)
    blendinfo = [
        f"{idx}",
        f"{distance:.6f}"]

    return blendinfo + gal2cat(blend.gal1) + gal2cat(blend.gal2)


def main(n_blend: int, datapath: str = 'data', seed: int = 42) -> None:
    n_blend = int(n_blend)
    
    cwd = pathlib.Path.cwd()
    
    outdir = cwd / f'output-s_{seed}-n_{n_blend}'
    if not outdir.exists():
        outdir.mkdir()

    datapath = cwd / datapath

    blender = Blender(datapath / 'candels.npy',
                      datapath / 'candels_seg.npy',
                      datapath / 'candels.csv',
                      magdiff=2,
                      raddiff=4,
                      seed=seed)

    blender.make_cut(blender.cat.mag > 18)
    blender.make_cut(blender.cat.mag < 23)
    blender.make_cut(blender.cat.galtype != 'irr')

    outcat = outdir / 'blend_cat.csv'
    with open(outcat, 'w') as f:
        output = csv.writer(f)
        output.writerow(CAT_HEADER)
        for blend_id in tqdm.trange(n_blend):
            blend = blender.next_blend()
            while blend is None:
                blend = blender.next_blend()
            output.writerow(blend2cat(blend, blend_id))
            save_img(blend, blend_id, outdir)


if __name__ == '__main__':
    import sys
    main(n_blend=int(sys.argv[1]), seed=int(sys.argv[2]))
