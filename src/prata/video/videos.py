import os
import numpy as np
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from natsort import natsorted
import pandas as pd
from loguru import logger

import typer
from tqdm.rich import tqdm

app = typer.Typer()


@app.command()
def cut(
    csv_path: Path = typer.Argument(..., help="csv path", exists=True, file_okay=True),
    video_path: Path = typer.Argument(
        ..., help="video base path", exists=True, dir_okay=True
    ),
    output_path: Path = typer.Argument(..., help="output base path"),
):
    if output_path.exists():
        inp = input("Existed, override? (y/n)")
        assert len(inp) == 1
        inp = inp.lower()
        if inp == "y":
            shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    # Read starting and ending time
    df = pd.read_csv(csv_path)
    for id, row in tqdm(df.iterrows(), total=len(df)):
        in_video = "{}".format(row["name"])
        out_video = "cut_{}.mp4".format(id)
        in_path = video_path / in_video
        assert in_path.exists()
        start = row.start
        end = row.end
        category = row.category
        out_path = output_path / category / out_video
        out_path.parent.mkdir(exist_ok=True, parents=True)
        cmd = f"ffmpeg -i '{in_path}' -ss {start} -to {end} -c copy {out_path}"
        logger.info(cmd)
        _ = subprocess.check_output(cmd, shell=True)


@app.command()
def toh264(
    input_path: Path = typer.Argument(..., help="input path", exists=True),
    output_path: Path = typer.Option("./output", help="Output path"),
    fps: int = typer.Option(10, help="FPS"),
    crf: int = typer.Option(18, help="CRF"),
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
        cur_outpath = output_path / input_path.name

        cmd = [
            "/usr/bin/ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_path.resolve().as_posix(),
            "-tune",
            "fastdecode",
            "-tune",
            "zerolatency",
            "-filter:v",
            f"fps={fps}",
            "-map",
            "0",
            "-c:v",
            "libx264",
            "-crf",
            f"{crf}",
            "-c:a",
            "copy",
            f"{cur_outpath.resolve().as_posix()}",
        ]
        # print(" ".join(cmd))
        subprocess.call(cmd)


@app.command()
def toh264_v2(
    input_path: Path = typer.Argument(..., help="input path", exists=True),
    output_path: Path = typer.Option("./output", help="Output path"),
    fps: int = typer.Option(10, help="FPS"),
    tmp_folder: Path = typer.Option("./h264tmp", help="temp folder"),
):
    inputs = []
    if input_path.is_file():
        inputs.append(input_path)
    elif input_path.is_dir():
        inputs = input_path.glob("*.[ma][pv][4i]")
        inputs = list(map(lambda x: x, inputs))
    pbar = tqdm(natsorted(inputs))
    output_path.mkdir(exist_ok=True, parents=True)
    tmp_folder.mkdir(exist_ok=True, parents=True)
    for input_path in pbar:
        assert input_path.exists(), f"{input_path} not exists!"
        pbar.set_description(f"{input_path.stem}")
        cur_outpath = output_path / input_path.name
        tmp_path = tmp_folder / "tmp.h264"
        if tmp_path.exists():
            os.remove(tmp_path)

        cmd = [
            "/usr/bin/ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            input_path.resolve().as_posix(),
            "-c",
            "copy",
            "-f",
            "h264",
            tmp_path.resolve().as_posix(),
        ]
        # print(" ".join(cmd))
        subprocess.call(cmd)
        cmd = [
            "/usr/bin/ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-r",
            str(fps),
            "-i",
            tmp_path.resolve().as_posix(),
            "-c",
            "copy",
            cur_outpath.resolve().as_posix(),
        ]
        subprocess.call(cmd)
    shutil.rmtree(tmp_folder)


if __name__ == "__main__":
    app()
