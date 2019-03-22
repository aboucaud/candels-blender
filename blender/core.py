from typing import List, NamedTuple

import numpy as np  # type: ignore


Galaxy = NamedTuple(
    "Galaxy",
    [
        ("cat_id", int),
        ("gal_id", int),
        ("mag", float),
        ("rad", float),
        ("z", float),
        ("type", str),
    ],
)

Blend = NamedTuple(
    "Blend",
    [
        ("img", np.ndarray),
        ("segmap", np.ndarray),
        ("gal1", Galaxy),
        ("gal2", Galaxy),
        ("shift", List[int]),
    ],
)
