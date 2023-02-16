from .kidnap import app as KidnapApp
from .meva import app as MevaApp

import typer

app = typer.Typer()
app.add_typer(KidnapApp, name="private")
app.add_typer(MevaApp, name="meva")
