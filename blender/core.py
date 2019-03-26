from typing import List, NamedTuple

from numpy import ndarray as Stamp  # pragma: no cover


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
        ("img", Stamp),
        ("segmap", Stamp),
        ("gal1", Galaxy),
        ("gal2", Galaxy),
        ("shift", List[int]),
    ],
)
