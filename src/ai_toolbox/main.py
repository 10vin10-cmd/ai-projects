import click

@click.group()
def cli():
	"""Main command group for the AI toolbox."""
	pass

@cli.command()
def hello():
	"""Print a greeting from the AI toolbox."""
	click.echo("Hello from the AI toolbox!")

if __name__ == "__main__":
	cli()