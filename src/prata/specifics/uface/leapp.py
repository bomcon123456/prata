import typer
from .validate_data import app as ValidateDataApp

app = typer.Typer()

app.add_typer(ValidateDataApp, name="dataval")
