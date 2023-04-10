import typer

from .facegen import app as FaceGenApp

app = typer.Typer()
app.add_typer(FaceGenApp, name="facegen")
