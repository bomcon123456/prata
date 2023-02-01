from genericpath import exists
import os
import typer
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil
import json

from .data_utils import list_categories
from prata.cvat.coco_utils import create_default_coco

app = typer.Typer()


@app.command()
def filter(
    input_path: Path = typer.Argument(
        ..., help="input path", exists=True, file_okay=True
    ),
    output_path: Path = typer.Argument(..., help="output path"),
):
    with open(input_path, "r") as f:
        d = json.load(f)
    result = create_default_coco(list_categories, list(range(len(list_categories))))
    print([x["name"] for x in d["categories"]])


if __name__ == "__main__":
    app()
