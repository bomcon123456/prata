import os
import shutil
import zipfile
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import typer
from natsort import natsorted
from tqdm.rich import tqdm

app = typer.Typer()


@app.command()
def zipfiles(
    input_path: Path = typer.Argument(..., help="input path", exists=True),
    output_path: Path = typer.Option("./output", help="Output path"),
    level: int = typer.Option(
        0,
        help="Level of file: 0: zip everything is this dir into one, 1: zip every folder into every zip",
    ),
):
    inputs = []
    dirs = []
    if level == 0:
        dirs = [input_path]
    elif level == 1:
        dirs = list(map(lambda x: input_path / x, os.listdir(input_path)))

    pbar = tqdm(natsorted(dirs))
    output_path.mkdir(exist_ok=True, parents=True)
    for input_path in pbar:
        files = natsorted(input_path.rglob("*"))
        with ZipFile("compressedtextstuff.zip", "w", zipfile.ZIP_DEFLATED) as myzip:
            myzip.write("testtext.txt")


@app.command()
def rename_folders(input_path: Path = typer.Argument(..., help="root input path")):
    folders = list(os.listdir(input_path))
    for folder in tqdm(folders):
        folder_name_new = folder.replace(".", "_").lower()
        folder_old = input_path / folder
        folder_new = input_path / folder_name_new
        shutil.move(folder_old, folder_new)


@app.command()
def move_file_keep_parent_name(
    input_path: Path = typer.Argument(..., help="input path"),
    output_path: Path = typer.Argument(..., help="Output path"),
):
    exts = set([".jpeg", ".jpg", ".png", ".mp4", ".avi"])
    files = list(input_path.rglob("*"))
    for file in tqdm(files):
        if file.suffix.lower() not in exts:
            continue
        prefix = file.parent.parent.name
        new_name = prefix + "_" + file.name
        new_path = output_path / file.parent.name / new_name
        new_path.parent.mkdir(exist_ok=True, parents=True)
        assert not new_path.exists(), f"{new_path} existed."
        shutil.copy2(file, new_path)


@app.command()
def move_all_file_to_folder(
    input_path: Path = typer.Argument(..., help="input path"),
    output_path: Path = typer.Argument(..., help="Output path"),
    check_exists: bool = typer.Option(False, help="chekc if exists"),
):
    files = list(input_path.rglob("*"))
    for file in tqdm(files):
        op = output_path / file.name
        if check_exists and op.exists():
            raise Exception(f"{file} is existed")
        shutil.move(file, op)


@app.command()
def csvs_to_parquet(
    csv_dir: Path = typer.Argument(..., help="input csvs"),
    output_path: Path = typer.Argument(..., help="output path"),
):
    csv_paths = csv_dir.rglob("*.csv")
    dfs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        dfs.append(df)
    df = pd.concat(dfs)
    del df["idx"]
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(output_path.as_posix())


@app.command()
def merge_csvs(
    csv_dir: Path = typer.Argument(..., help="input csvs", dir_okay=True, exists=True),
    output_path: Path = typer.Argument(..., help="output path"),
    column_names: str = typer.Option(None, help="column name"),
):
    csv_paths = list(csv_dir.rglob("*.csv"))
    df = pd.concat(map(pd.read_csv, csv_paths), ignore_index=True)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    if column_names is not None:
        n_cols = len(df.axes[1])
        column_names = list(map(lambda x: x.strip(), column_names.split(",")))
        assert len(column_names) == n_cols
        df.columns = column_names
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    app()
