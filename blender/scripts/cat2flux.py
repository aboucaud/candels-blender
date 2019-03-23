from pathlib import Path

import click
import numpy as np
import pandas as pd


def mag2flux(mag, zp):
    "Convert from magnitude to flux"
    return 10 ** (-0.4 * (mag - zp))


@click.command("convert")
@click.argument('image_dir', type=click.Path(exists=True))
@click.option('--zeropoint', default=25.96,
               help="Magnitude zero point for the conversion to flux")
def main(image_dir, zeropoint):
    """Create an array with the flux of the blended galaxies"""
    path = Path.cwd() / image_dir

    for prefix in ["train", "test"]:
        catalog = path / f'{prefix}_catalogue.csv'
        output_file = path / f'{prefix}_flux.npy'

        df = pd.read_csv(catalog)
        g1_flux = mag2flux(df.g1_mag.values, zp=zeropoint)
        g2_flux = mag2flux(df.g2_mag.values, zp=zeropoint)

        fluxes = [g1_flux[:, None], g2_flux[:, None]]

        flux_array = np.concatenate(fluxes, axis=-1)

        np.save(output_file, flux_array)

        click.echo(f'=> {output_file} created')


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter