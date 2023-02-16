from genericpath import exists
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


if __name__ == "__main__":
    app()
