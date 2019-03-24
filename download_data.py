
"""
Download script for data stored remotely
----------------------------------------

author: Alexandre Boucaud <aboucaud@apc.in2p3.fr>
license: BSD (3-clause)

"""
import os
import sys
import tarfile

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

import click

ZENODO_URL = "https://zenodo.org/record"
RECORD_NUMBER = 2604740
FILES = [
    "candels-blender-data.tar.gz",
]


@click.command()
@click.option(
    "-o",
    "--output_dir",
    type=click.Path(exists=False, dir_okay=True),
    default="data",
    show_default=True,
    help="Destination directory to download the data, will be created if not existing"
)
@click.option(
    "--delete",
    is_flag=True,
    help="Delete archives once extracted"
)
def main(output_dir, delete):
    urls = [
        f"{ZENODO_URL}/{RECORD_NUMBER}/files/{filename}"
        for filename in FILES
    ]

    if not os.path.exists(output_dir):
        click.echo(f"Creating directory {output_dir}")
        os.mkdir(output_dir)

    for url, filename in zip(urls, FILES):
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            click.echo(f"{filename} already downloaded.")
            continue

        click.echo(f"Downloading from {url} ...")
        urlretrieve(url, filename=output_file)
        click.echo(f"=> File saved as {output_file}")

        if filename.endswith("tar.gz"):
            click.echo("Extracting tarball..")
            with tarfile.open(output_file, "r:gz") as f:
                f.extractall(output_dir)
            click.echo("Done.")

            if delete:
                os.remove(output_file)
                click.echo(f"=> File {output_file} removed.")


if __name__ == '__main__':
    main()
