from .kidnap import app as KidnapApp
from .meva import app as MevaApp
from .vidat import app as VidatApp

import typer

app = typer.Typer()
app.add_typer(KidnapApp, name="private")
app.add_typer(MevaApp, name="meva")
app.add_typer(VidatApp, name="vidat")
