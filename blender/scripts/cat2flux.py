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
@click.option('--train', 'prefix', flag_value='train', default=True,
              help="Apply to train images")
@click.option('--test', 'prefix', flag_value='test',
              help="Apply to test images")
def main(image_dir, zeropoint, prefix):
    """Create an array with the flux of the blended galaxies"""
    path = Path.cwd() / image_dir

    catalog = path / f'{prefix}_blend_cat.csv'
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