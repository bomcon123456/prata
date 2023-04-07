import typer
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from typing import *
from pathlib import Path
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path.as_posix())


@app.command()
def histogram_csv(
    input_path: Path = typer.Argument(..., help="input path"),
    output_path: Path = typer.Argument(..., help="output path"),
    column_names: List[str] = typer.Argument(..., help="column to plot"),
    bins: List[int] = typer.Option([10], help="bins"),
    xlabel: str = typer.Option("", help="xlabel"),
    ylabel: str = typer.Option("", help="ylabel"),
    title: str = typer.Option("", help="title"),
):
    df = pd.read_csv(input_path)
    my_colors = [
        (x / 10.0, x / 20.0, 0.75) for x in range(len(column_names))
    ]  # <-- Quick gradient example along the Red/Green dimensions.

    for i, column_name in enumerate(column_names):
        if len(bins) == 1:
            print(bins)
            df[column_name].hist(
                bins=bins[0], grid=False, figsize=(12, 8), alpha=0.5, label=column_name
            )
        else:
            df[f"{column_name}_bins"] = pd.cut(df[column_name], bins)
            df[f"{column_name}_bins"].value_counts().plot.bar(
                alpha=0.5, label=column_name, figsize=(12, 8), color=my_colors[i]
            )

    plt.title(title)

    # Set x-axis label
    plt.xlabel(xlabel, labelpad=20, weight="bold", size=12)

    # Set y-axis label
    plt.ylabel(ylabel, labelpad=20, weight="bold", size=12)

    plt.legend()
    plt.savefig(output_path.as_posix())


@app.command()
def bar_csv(
    input_path: Path = typer.Argument(..., help="input path"),
    output_path: Path = typer.Argument(..., help="output path"),
    column_names: List[str] = typer.Argument(..., help="column to plot"),
    xlabel: str = typer.Option("", help="xlabel"),
    ylabel: str = typer.Option("", help="ylabel"),
    title: str = typer.Option("", help="title"),
):
    df = pd.read_csv(input_path)
    for i, column_name in enumerate(column_names):
        df[column_name].value_counts().plot.bar(
            alpha=0.5, label=column_name, figsize=(12, 8)
        )

    plt.title(title)

    # Set x-axis label
    plt.xlabel(xlabel, labelpad=20, weight="bold", size=12)

    # Set y-axis label
    plt.ylabel(ylabel, labelpad=20, weight="bold", size=12)

    plt.legend()
    plt.savefig(output_path.as_posix())


@app.command()
def plot_widerface(
    input_path: Path = typer.Argument(..., help="input path"),
    output_path: Path = typer.Argument(..., help="output path"),
    samples: int = typer.Option(100, help="#samples"),
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


@app.command()
def plot_facegen_parquet(
    input_path: Path = typer.Argument(..., help="input path"),
    output_path: Path = typer.Argument(..., help="output path"),
    iqa_threshold: int = typer.Option(60, help="iqa threshold"),
    xlabel: str = typer.Option("", help="xlabel"),
    ylabel: str = typer.Option("", help="ylabel"),
    title: str = typer.Option("", help="title"),
):

    df = pd.read_csv(input_path)
    filtered_by_iqa_df = df[df["iqa"] > iqa_threshold & df["iqa"] < 100]["iqa"]
    column_names = ["profile_left", "profile_right", "profile_up", "profile_down"]
    for i, column_name in enumerate(column_names):
        filtered_by_iqa_df[column_name].value_counts().plot.bar(
            alpha=0.5, label=column_name, figsize=(12, 8)
        )

    plt.title(title)

    # Set x-axis label
    plt.xlabel(xlabel, labelpad=20, weight="bold", size=12)

    # Set y-axis label
    plt.ylabel(ylabel, labelpad=20, weight="bold", size=12)

    plt.legend()
    plt.savefig(output_path.as_posix())


if __name__ == "__main__":
    app()
