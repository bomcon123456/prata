import typer
from .validate_data import app as ValidateDataApp
from .g3 import app as G3App

app = typer.Typer()

app.add_typer(ValidateDataApp, name="dataval")
app.add_typer(G3App, name="g3")
