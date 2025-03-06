import click
from src.recommendation import cli as recommendation_cli
from src.convolution import cli as convolution_cli

@click.group()
def cli():
    """DataSnafu CLI - Machine Learning Operations Tool"""
    pass

# Add subcommands from different modules
cli.add_command(recommendation_cli, "recommendation")
cli.add_command(convolution_cli, "convolution")

@cli.command()
@click.option('--debug/--no-debug', default=False, help='Enable debug mode')
@click.option('--config', type=click.Path(), help='Path to config file')
def init(debug, config):
    """Initialize the project configuration"""
    click.echo(f"Initializing project with debug={debug}")
    if config:
        click.echo(f"Using config file: {config}")

@cli.command()
def version():
    """Show the version information"""
    click.echo("DataSnafu v0.1.0")

if __name__ == '__main__':
    cli()