"""
candels-blender main command line interface

\b
select one of the following actions:
- `produce`: create the blends, masks and catalogues
- `concatenate`: arrange the blends products into files
- `convert`: create the flux table
"""
import click

from blender.scripts import produce_blends
from blender.scripts import concatenate_blends
from blender.scripts import cat2flux


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=__doc__,
)
def cli():
    pass


cli.add_command(produce_blends.main)
cli.add_command(concatenate_blends.main)
cli.add_command(cat2flux.main)


if __name__ == "__main__":
    cli()