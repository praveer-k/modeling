import click

from modeling.config import logger
from modeling.llm.__main__ import cli as llm_cli
from modeling.docs.__main__ import cli as docs_cli
from modeling.convolution.__main__ import cli as convolution_cli
from modeling.recommendation.__main__ import cli as recommendation_cli


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


common_options = [
    click.option(
        "--debug/--no-debug", "-d/-D", default=False, help="Enable debug mode"
    ),
    click.option("--config", "-c", type=click.Path(), help="Path to config file"),
]


@click.group()
@add_options(common_options)
@click.pass_context
def cli(ctx: click.Context, debug: bool, config: str):
    """Modeling CLI - Machine Learning Operations Tool"""
    ctx.ensure_object(dict)
    # Setup logging
    ctx.obj["LOGGER"] = logger
    # Store debug and config in context
    ctx.obj["DEBUG"] = debug
    ctx.obj["CONFIG"] = config
    if debug:
        logger.setLevel("DEBUG")
        logger.debug("Debug mode enabled")
        if config:
            logger.debug(f"Using config file: {config}")


cli.add_command(recommendation_cli, "recommendation")
cli.add_command(convolution_cli, "convolution")
cli.add_command(llm_cli, "llm")
cli.add_command(docs_cli, "docs")


@cli.command()
def version():
    """Show the version information"""
    click.echo("Modeling v0.1.0")


if __name__ == "__main__":
    cli(obj={})
