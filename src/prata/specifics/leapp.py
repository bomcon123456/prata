import typer
from .kidnap import app as KidnapApp
from .uface import UFaceApp
from .yolov5face import YoloApp
from .research import ResearchApp

app = typer.Typer()
app.add_typer(KidnapApp, name="kidnap")
app.add_typer(UFaceApp, name="uface")
app.add_typer(YoloApp, name="yolo")
app.add_typer(ResearchApp, name="research")