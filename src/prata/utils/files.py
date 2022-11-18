import os
import zipfile
from pathlib import Path
from zipfile import ZipFile

import typer
from natsort import natsorted
from tqdm.rich import tqdm

app = typer.Typer()


@app.command()
def zipfiles(
    input_path: Path = typer.Argument(..., help="input path", exists=True),
    output_path: Path = typer.Option("./output", help="Output path"),
    level: int = typer.Option(
        0,
        help="Level of file: 0: zip everything is this dir into one, 1: zip every folder into every zip",
    ),
):
    inputs = []
    dirs = []
    if level == 0:
        dirs = [input_path]
    elif level ==1:
        dirs = list(map(lambda x: input_path/x, os.listdir(input_path)))
        
    pbar = tqdm(natsorted(dirs))
    output_path.mkdir(exist_ok=True, parents=True)
    for input_path in pbar:
        files = natsorted(input_path.rglob("*"))
        with ZipFile("compressedtextstuff.zip", "w", zipfile.ZIP_DEFLATED) as myzip:
            myzip.write("testtext.txt")


if __name__ == "__main__":
    app()
