from typing import List, Sequence

import numpy as np  # type: ignore

from blender.core import Blend, Galaxy

CATALOG_HEADER: Sequence[str] = (
    "id",
    "distance",
    "shift_x",
    "shift_y",
    "g1_id",
    "g1_mag",
    "g1_rad",
    "g1_z",
    "g1_type",
    "g2_id",
    "g2_mag",
    "g2_rad",
    "g2_z",
    "g2_type",
)


def gal2cat(gal: Galaxy) -> List[str]:
    """
    id rad mag z type
    """
    galinfo: List[str] = [
        f"{gal.gal_id}",
        f"{gal.mag:.6f}",
        f"{gal.rad:.6f}",
        f"{gal.z:.6f}",
        f"{gal.type}",
    ]

    return galinfo


def blend2cat(blend: Blend, idx: int) -> List[str]:
    """
    blend_id distance gal1_id gal1_mag gal1_rad gal1_z gal1_type gal2_id gal2_mag gal2_rad gal2_z gal2_type  # noqa
    """
    distance = np.hypot(*blend.shift)
    blendinfo: List[str] = [
        f"{idx}",
        f"{distance:.6f}",
        f"{blend.shift[0]}",
        f"{blend.shift[1]}",
    ]

    return blendinfo + gal2cat(blend.gal1) + gal2cat(blend.gal2)
