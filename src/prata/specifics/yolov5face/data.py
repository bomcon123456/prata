from genericpath import exists
import os
import typer
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil

from .data_utils import oneclassify_onefolder

app = typer.Typer()

@app.command()
def oneclassify(
    input_path: Path = typer.Argument(...,help="input path", exists=True, dir_okay=True),
    output_path: Path = typer.Argument(...,help="output path"),
    level: int = typer.Option(0, help="0->that folder only, 1->everysubfolder"),
):
    if level == 0:
        oneclassify_onefolder(input_path, output_path)
    else:
        folders = os.listdir(input_path)
        for folder in tqdm(folders):
            inp = input_path / folder
            out = output_path / folder
            oneclassify_onefolder(inp, out)
    
    

if __name__ == "__main__":
    app()