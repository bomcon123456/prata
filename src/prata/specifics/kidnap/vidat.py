import typer
from tqdm import tqdm
from natsort import natsorted
import os
import shutil
from pathlib import Path

from prata.specifics.kidnap.vidat_utils import cut_one_video

app = typer.Typer()


@app.command()
def main(
    video_path: Path = typer.Argument(..., help="Video path", exists=True),
    annotation_path: Path = typer.Argument(..., help="Annotation path", exists=True),
    output_path: Path = typer.Argument(..., help="output path"),
):
    video_paths = []
    if video_path.is_file():
        video_paths = [video_path]
    else:
        video_paths = list(video_path.rglob("*[ma][pvk][4iv]"))
    for video_path in tqdm(natsorted(video_paths)):
        video_name = video_path.stem
        video_ann_path = annotation_path / f"{video_name}.json"
        assert video_ann_path.exists(), f"Annotation path: {video_ann_path} not exists"
        cut_one_video(video_path, video_ann_path, output_path)


if __name__ == "__main__":
    app()
