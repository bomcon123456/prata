import typer
from prata.video import FramesApp, VideosApp
from prata.cvat import CvatAPIApp
from prata.specifics import KidnapApp, UFaceApp
from prata.utils import app as UtilsApp
from prata.plot import PlotApp

app = typer.Typer()
app.add_typer(FramesApp, name="frames")
app.add_typer(VideosApp, name="videos")
app.add_typer(CvatAPIApp, name="cvat")
app.add_typer(KidnapApp, name="kidnap")
app.add_typer(UFaceApp, name="uface")
app.add_typer(UtilsApp, name="utils")
app.add_typer(PlotApp, name="plot")
