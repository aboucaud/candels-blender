from astropy.visualization import ImageNormalize
from astropy.visualization import MinMaxInterval
from astropy.visualization import LogStretch
from astropy.visualization import AsinhStretch

from blender.core import Stamp


def asin_stretch_norm(images: Stamp):
    return ImageNormalize(
        images,
        interval=MinMaxInterval(),
        stretch=AsinhStretch(),
    )

def log_stretch_norm(images: Stamp):
    return ImageNormalize(
        images,
        interval=MinMaxInterval(),
        stretch=LogStretch(),
    )
