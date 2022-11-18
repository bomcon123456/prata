import typer
import os
import shutil
from pathlib import Path

app = typer.Typer()


@app.command()
def move(
    input_path: Path = typer.Argument(..., help="Path to input"),
    output_path: Path = typer.Option("./out", help="output path"),
):
    val_folders = [
        "kidnap_20220816103233-20220816113000_2",
        "kidnap_20220816104513-20220816114500_0",
        "kidnap_20220816103053-20220816113000_1_kidnap",
        "kidnap_20220816103249-20220816113000_1_kidnap",
        "kidnap_20220816103233-20220816113000_0",
        "kidnap_20220816104513-20220816114500_1_kidnap",
        "kidnap_20220816103249-20220816113000_4",
        "22_20220816105714_20220816113000_8568764",
        "kidnap_20220816104513-20220816114500_2",
        "kidnap_20220816103249-20220816113000_0",
        "kidnap_20220816103249-20220816113000_2",
        "kidnap_20220816103053-20220816113000_2",
        "kidnap_20220816103233-20220816113000_4",
        "kidnap_20220816103233-20220816113000_1_kidnap",
        "kidnap_20220816103233-20220816113000_3_kidnap",
        "kidnap_20220816103249-20220816113000_3_kidnap",
    ]
    val_path = output_path / "val"
    train_path = output_path / "train"
    (output_path / "train").mkdir(exist_ok=True, parents=True)
    (output_path / "val").mkdir(exist_ok=True, parents=True)
    scenes = set(os.listdir(input_path))
    for vf in val_folders:
        p = input_path / vf
        assert p.exists()
        shutil.copytree(p, val_path / vf)
        scenes.remove(vf)
    for tf in scenes:
        p = input_path / tf
        assert p.exists()
        shutil.copytree(p, train_path / tf)


if __name__ == "__main__":
    app()
