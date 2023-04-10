# from .fiftyone import app as FiftyOneApp
import typer

from .files import app as FileApp

app = typer.Typer()
app.add_typer(FileApp, name="files")
# app.add_typer(FiftyOneApp, name="fiftyone")
