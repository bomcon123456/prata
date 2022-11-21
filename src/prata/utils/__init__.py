from .files import app as FileApp
import typer

app = typer.Typer()
app.add_typer(FileApp, name="files")