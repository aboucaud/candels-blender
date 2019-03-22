import logging
from typing import List, Tuple, Union, Optional
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from numpy import ndarray as Stamp  # pragma: no cover
from numpy.random import RandomState

from blender.core import Galaxy, Blend
from blender.segmap import normalize_segmap
from blender.segmap import mask_out_pixels

PathType = Union[Path, str]

class BlendShiftError(Exception):
    pass


class BlendMissingTestError(Exception):
    pass


class Blender:
    img_dtype = np.float32
    seg_dtype = np.uint8

    def __init__(self, imgpath: PathType, segpath: PathType, catpath: PathType,
                 train_test_ratio: float = 0.2,
                 magdiff: int = 2, raddiff: int = 4, seed: int = 42) -> None:
        self.data = np.load(imgpath).astype(self.img_dtype, copy=False)
        self.seg = np.load(segpath).astype(self.seg_dtype, copy=False)
        self.cat = pd.read_csv(catpath)
        self.tt_ratio = np.clip(train_test_ratio, 0, 1)
        self.magdiff = magdiff
        self.raddiff = raddiff
        self.rng = RandomState(seed=seed)
        self.img_size = self.data.shape[-1]

        self.assign_train_test()

    @property
    def n_gal(self) -> int:
        return len(self.data)

    def assign_train_test(self) -> None:
        randomized_indices = self.rng.permutation(self.n_gal)
        split_idx = int(self.n_gal * self.tt_ratio)
        test, train = np.split(randomized_indices, [split_idx])
        self.train_idx = train
        self.test_idx = test

    def galaxy(self, idx: int) -> Galaxy:
        galfields = ["ID", "mag", "radius", "z", "galtype"]
        return Galaxy(idx, *self.cat.iloc[idx][galfields])

    def original_stamp(self,
                       gal: Galaxy,
                       norm_segmap: bool = False) -> Tuple[Stamp, Stamp]:
        gal_id = gal.cat_id

        img = self.data[gal_id].copy()
        seg = self.seg[gal_id].copy()

        if norm_segmap:
            seg = normalize_segmap(seg)

        return img, seg

    def masked_stamp(self, gal: Galaxy) -> Tuple[Stamp, Stamp]:
        gal_id = gal.cat_id

        img = self.data[gal_id].copy()
        seg = self.seg[gal_id].copy()
        segval = seg[64, 64]

        masked_img = mask_out_pixels(img, seg, segval)

        return masked_img, self.clean_seg(gal_id)

    def make_cut(self, logic) -> None:
        self.data = self.data[logic]
        self.seg = self.seg[logic]
        self.cat = self.cat[logic].reset_index(drop=True)

        self.assign_train_test()

    def clean_seg(self, idx: int) -> Stamp:
        """Return the segmentation contours of the central object only"""
        return np.where(self.seg[idx] == self.seg[idx, 64, 64],
                        1, 0).astype(self.seg_dtype)

    def pad(self, array) -> Stamp:
        return np.pad(array, self.img_size // 2, mode="constant")

    def crop(self, array) -> Stamp:
        padding = self.img_size // 2
        cut = slice(padding, self.img_size + padding)
        return array[cut, cut]

    def shift(self, array, coords: List[int]) -> Stamp:
        dy, dx = coords
        array = self.pad(array)
        array = np.roll(np.roll(array, dx, axis=1), dy, axis=0)
        array = self.crop(array)
        return array

    def blend(self, gal1: Galaxy, gal2: Galaxy, masked: bool = True) -> Blend:
        if masked:
            img, seg = self.masked_stamp(gal1)
            img2, seg2 = self.masked_stamp(gal2)
        else:
            img, seg = self.original_stamp(gal1, norm_segmap=True)
            img2, seg2 = self.original_stamp(gal2, norm_segmap=True)

        coords = self.random_shift(gal1, gal2)
        if coords is None:
            raise BlendShiftError("Cannot find proper displacement")

        img2 = self.shift(img2, coords)
        seg2 = self.shift(seg2, coords)

        img_cube = np.concatenate([img[..., None], img2[..., None]], axis=-1)
        seg_cube = np.concatenate([seg[None, ...], seg2[None, ...]])

        assert img_cube.dtype == self.img_dtype
        assert seg_cube.dtype == self.seg_dtype

        return Blend(
            img=img_cube,
            segmap=seg_cube,
            gal1=gal1,
            gal2=gal2,
            shift=coords
        )

    def random_galaxy(self, from_test: bool = False) -> Galaxy:
        "Pick a random galaxy from the catalog"
        if from_test:
            try:
                gal = self.galaxy(self.rng.choice(self.test_idx))
            except ValueError:
                raise BlendMissingTestError(
                    "The test set has not been specified. "
                    "Initialize the blender with a non-zero "
                    "`train_test_ratio`.")

        else:
            gal = self.galaxy(self.rng.choice(self.train_idx))

        return gal

    def random_pair(self, from_test: bool = False) -> Tuple[Galaxy, Galaxy]:
        "Pick a random pair of galaxies with specific flux constrains"
        gal1 = self.random_galaxy(from_test)
        gal2 = self.random_galaxy(from_test)
        while not np.abs(gal1.mag - gal2.mag) < self.magdiff:
            gal2 = self.random_galaxy()

        return gal1, gal2

    def random_shift(self, gal1: Galaxy, gal2: Galaxy) -> Optional[List[int]]:
        # Min radius has to be the biggest of both effective radii
        rad_min = max(gal1.rad, gal2.rad)
        # Max radius is defined as a factor of the smallest effective radius
        rad_max = min(min(gal1.rad, gal2.rad) * self.raddiff,
                      self.img_size // 2)

        if rad_min >= rad_max:
            rad_min = 0.8 * rad_max

        tryouts = 25
        coords = [0, 0]
        while not (rad_min <= np.hypot(*coords) <= rad_max):
            if tryouts <= 0:
                return None
            coords = self.rng.randint(-rad_max, rad_max, size=2).tolist()
            tryouts -= 1

        return coords

    def next_blend(self,
                   from_test: bool = False,
                   masked: bool = True) -> Optional[Blend]:
        gal1, gal2 = self.random_pair(from_test)

        try:
            blend = self.blend(gal1, gal2, masked=masked)
        except BlendShiftError:
            logger = logging.getLogger(__name__)
            logger.info(
                f"Issue while blending galaxies {gal1.gal_id} and "
                f"{gal2.gal_id}")
            return None

        return blend

    def plot_data(self, idx: int) -> None:
        import matplotlib.pyplot as plt

        title_list = [
            "Galaxy stamp",
            "CANDELS segmentation map",
            "Cleaned segmentation map"
        ]

        fig, axes = plt.subplots(1, 3, figsize=(12, 8), tight_layout=True)

        axes[0].imshow(self.data[idx], origin="lower")
        axes[1].imshow(normalize_segmap(self.seg[idx]), origin="lower")
        axes[2].imshow(self.clean_seg(idx), origin="lower")

        for ax, title in zip(axes, title_list):
            ax.set_axis_off()
            ax.set_title(title)

        return fig

    def plot_blend(self, idx1: int, idx2: int):
        import matplotlib.pyplot as plt
        import astropy.visualization as viz

        g1 = self.galaxy(idx1)
        g2 = self.galaxy(idx2)
        blend = self.blend(g1, g2)
        imglist = [
            blend.img,
            blend.segmap[0],
            blend.segmap[1],
            blend.segmap.sum(axis=0)]
        titlelist = [
            f"blend image {g1.gal_id} - {g2.gal_id}",
            f"{g1.type} - mag:{g1.mag:.2f} - rad:{g1.rad:.2f}",
            f"{g2.type} - mag:{g2.mag:.2f} - rad:{g2.rad:.2f}",
            f"blend segmap {g1.cat_id} - {g2.cat_id}"]
        norm = viz.ImageNormalize(blend.img,
                                  interval=viz.MinMaxInterval(),
                                  stretch=viz.SqrtStretch())
        fig, axes = plt.subplots(1, 4, figsize=(16, 8), tight_layout=True)
        for i, image in enumerate(imglist):
            if i == 0:
                axes[i].imshow(image, origin="lower", norm=norm)
            else:
                axes[i].imshow(image, origin="lower")
            axes[i].set_axis_off()
            axes[i].set_title(titlelist[i])

        return fig
