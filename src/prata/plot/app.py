import typer
import cv2
from tqdm import tqdm
from typing import *
from pathlib import Path
import os
import matplotlib.pyplot as plt

from .utils import read_widerface

app = typer.Typer()


@app.command()
def pie_dist(
    input_path: Path = typer.Argument(..., help="input path"),
    output_path: Path = typer.Argument(..., help="output path"),
    ignored_folder: Optional[List[str]] = typer.Option(
        None, help="folder name to ignore"
    ),
):
    folders = os.listdir(input_path)
    d = {}
    for folder in folders:
        if folder in ignored_folder:
            continue
        p = input_path / folder
        d[folder] = len(os.listdir(p))
    fig1, ax1 = plt.subplots()
    ax1.pie(
        list(d.values()),
        labels=(d.keys()),
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title(f"Total images: {sum(d.values())}")
    plt.savefig(output_path.as_posix())

@app.command()
def plot_widerface(
    input_path: Path = typer.Argument(..., help="input path"),
    output_path: Path = typer.Argument(..., help="output path"),
    samples: int = typer.Option(100, help="#samples")
):
    txtfiles = list(input_path.rglob("*.txt"))
    if samples > len(txtfiles) or samples == -1:
        pass
    else:
        txtfiles = txtfiles[:samples]
    output_path.mkdir(exist_ok=True, parents=True)
    for txtfile in tqdm(txtfiles):
        img = read_widerface(txtfile)
        op = output_path / (txtfile.stem + ".jpg")
        cv2.imwrite(op.as_posix(), img)

if __name__ == "__main__":
    app()
