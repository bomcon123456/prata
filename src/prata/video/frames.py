import base64
import os
import shutil
import subprocess
import sys
import zipfile
from enum import Enum
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import typer
from genericpath import exists
from loguru import logger
from natsort import natsorted
from PIL import Image
from tqdm.rich import tqdm

app = typer.Typer()


@app.command()
def video2frames(
    input_path: Path = typer.Argument(..., help="input path", exists=True),
    output_path: Path = typer.Option("./output", help="Output path"),
    fps: int = typer.Option(2, help="FPS"),
):
    inputs = []
    if input_path.is_file():
        inputs.append(input_path)
    elif input_path.is_dir():
        inputs = input_path.glob("*.[ma][pv][4i]")
        inputs = list(map(lambda x: x, inputs))
    pbar = tqdm(natsorted(inputs))
    output_path.mkdir(exist_ok=True, parents=True)
    for input_path in pbar:
        assert input_path.exists(), f"{input_path} not exists!"
        pbar.set_description(f"{input_path.stem}")
        cur_outpath = output_path / input_path.stem
        cur_outpath.mkdir(exist_ok=True, parents=True)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_path.resolve().as_posix(),
            "-vf",
            f"fps={fps}",
            f"{cur_outpath.resolve().as_posix()}/%05d.jpg",
        ]
        # print(" ".join(cmd))
        # exit()
        subprocess.call(cmd)


@app.command()
def video2interval(
    input_path: Path = typer.Argument(..., help="input path", exists=True),
    output_path: Path = typer.Option("./output", help="Output path"),
    interval: float = typer.Option(
        15, help="Minute interval (extract every X minutes)"
    ),
    frames: int = typer.Option(
        60, help="Number of frames per interval (with the new fps)"
    ),
    fps: int = typer.Option(2, help="FPS"),
    seperate_folders: bool = typer.Option(
        False, help="Create newfolder for each interval?"
    ),
):
    inputs = []

    if input_path.is_file():
        inputs.append(input_path)
    elif input_path.is_dir():
        inputs = input_path.glob("*.[ma][pv][4i]")
        inputs = list(map(lambda x: x, inputs))
    pbar = tqdm(natsorted(inputs))
    # pbar = natsorted(inputs)
    output_path.mkdir(exist_ok=True, parents=True)
    interval_second = interval * 60
    for input_path in pbar:
        assert input_path.exists(), f"{input_path} not exists!"
        pbar.set_description(f"{input_path.stem}")
        cur_outpath = output_path / input_path.stem
        cur_outpath.mkdir(exist_ok=True, parents=True)

        command = f"ffprobe -i '{input_path.resolve().as_posix()}' -show_format -v quiet | sed -n 's/duration=//p'"
        video_seconds = int(
            float(
                subprocess.check_output(["bash", "-c", command])
                .decode("ascii")
                .replace("\n", "")
            )
        )
        intervals = np.arange(0, video_seconds, interval_second)
        for i, start_second in enumerate(intervals):
            if seperate_folders:
                final_outpath = cur_outpath / str(i)
                final_outpath.mkdir(exist_ok=True, parents=True)
                start_number = 0
            else:
                final_outpath = cur_outpath
                start_number = i
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(start_second),
                "-i",
                input_path.resolve().as_posix(),
                "-vframes",
                f"{frames+1}",
                "-vf",
                f"fps={fps}",
                "-start_number",
                f"{start_number}",
                "-q:v",
                "1",
                "-qmin",
                "1",
                "-qmax",
                "1",
                f"{final_outpath.resolve().as_posix()}/%05d.jpg",
            ]
            # print(" ".join(cmd))
            # exit()
            subprocess.call(cmd)


@app.command()
def filter_img_by_size(
    input_path: Path = typer.Argument(..., help="input path"),
    output_path: Path = typer.Argument(..., help="Output path"),
    min_width: int = typer.Option(200, help="min_width"),
    min_height: int = typer.Option(200, help="min_height"),
):
    exts = set([".jpeg", ".jpg", ".png"])
    files = list(input_path.rglob("*"))
    for file in tqdm(files):
        if file.suffix.lower() not in exts:
            continue
        img = cv2.imread(file.as_posix())
        h, w, c = img.shape
        if h < min_height or w < min_width:
            continue
        p = file.relative_to(input_path)
        op = output_path / p
        op.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(file, op)


@app.command()
def zipimages_to_thumbnail(
    zip_path: Path = typer.Argument(..., help="zip path"),
    out_path: Path = typer.Argument(..., help="out path"),
):
    if zip_path.is_dir():
        zip_paths = zip_path.rglob("*.zip")
    else:
        zip_paths = [zip_path]
    pbar = tqdm(zip_paths)
    for zip_path in pbar:
        pbar.set_description(f"{zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            for file_name in tqdm(list(zip_file.namelist())):
                if not file_name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".gif")
                ):
                    continue
                with zip_file.open(file_name) as my_file:
                    image_bytes = my_file.read()
                    with Image.open(BytesIO(image_bytes)) as img:
                        img.thumbnail((50, 50))
                        p = out_path / zip_path.stem / Path(file_name).name
                        p.parent.mkdir(exist_ok=True, parents=True)
                        img.save(p.as_posix())


if __name__ == "__main__":
    app()
