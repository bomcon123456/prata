import typer
import cv2
import numpy as np
import json
from tqdm import tqdm
from natsort import natsorted
import os
import shutil
from pathlib import Path

from .meva_file_utils import parse_yaml, Activity, NumpyEncoder
from .meva_vid_utils import crop_from_activities, crop_from_activity_cached

app = typer.Typer()


@app.command()
def cut_from_one_yaml(
    yaml_path: Path = typer.Argument(
        ..., help="YAML Path", file_okay=True, exists=True
    ),
    video_path: Path = typer.Argument(
        ..., help="Video path", file_okay=True, exists=True
    ),
    output_path: Path = typer.Argument(..., help="Output path"),
    filter_labels: bool = typer.Option(
        True, help="Take only labels that support kidnap"
    ),
    cached: bool = typer.Option(True, help="Cache videoframes"),
    engine: str = typer.Option("opencv", help="how to parse video"),
):
    filter_set = None
    if filter_labels:
        from .meva_assets import list_activities

        filter_set = list_activities
    activities = parse_yaml(yaml_path)
    
    crop_from_activities(video_path, activities, engine, cached, output_path, filter_set)
    return activities


@app.command()
def cut_from_yamls(
    yaml_basepath: Path = typer.Argument(
        ..., help="YAML Path", dir_okay=True, exists=True
    ),
    video_basepath: Path = typer.Argument(
        ..., help="Video path", dir_okay=True, exists=True
    ),
    output_path: Path = typer.Argument(..., help="Output path"),
    filter_labels: bool = typer.Option(
        True, help="Take only labels that support kidnap"
    ),
    cached: bool = typer.Option(True, help="Cache videoframes"),
):
    yaml_files = natsorted(list(yaml_basepath.rglob("*.activities.yml")))
    for yaml_file in tqdm(yaml_files):
        relative_parent = yaml_file.relative_to(yaml_basepath).parent
        yaml_filename = yaml_file.stem
        filenamebase = ".".join(yaml_filename.split(".")[:-1])
        video_folder = video_basepath / relative_parent
        video_files = list(video_folder.glob(f"{filenamebase}*.avi"))
        assert len(video_files) <= 1, f"Many files with similar names: {video_files}"
        if len(video_files) == 0:
            print(f"{yaml_filename} doesnt have video!")
            continue
        video_file = video_files[0]

        cut_from_one_yaml(yaml_file, video_file, output_path, filter_labels, cached)


if __name__ == "__main__":
    app()
