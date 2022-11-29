import typer
from .data import app as DataApp

app = typer.Typer()
app.add_typer(DataApp, name="data")
