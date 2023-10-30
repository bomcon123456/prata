# from .fiftyone import app as FiftyOneApp
import typer

from .files import app as FileApp
from .paths import CACHE_PATH, generate_cachepath_from_path, get_filelist_and_cache

app = typer.Typer()
app.add_typer(FileApp, name="files")
# app.add_typer(FiftyOneApp, name="fiftyone")
