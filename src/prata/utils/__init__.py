from .files import app as FileApp

# from .fiftyone import app as FiftyOneApp
import typer

app = typer.Typer()
app.add_typer(FileApp, name="files")
# app.add_typer(FiftyOneApp, name="fiftyone")
