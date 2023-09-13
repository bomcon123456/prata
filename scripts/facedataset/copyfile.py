from pathlib import Path
from tqdm import tqdm
import pandas as pd
import shutil
import typer

app = typer.Typer()

@app.command()
def area(
    csv_path: Path = typer.Argument(..., help="csv path", file_okay=True, exists=True),
    image_dir: Path = typer.Argument(..., help="Image Dir", dir_okay=True, exists=True),
    out_path: Path = typer.Argument(..., help="out path"),
):
    df = pd.read_csv(csv_path)
    print(max(df["area"]))
    expected_row = len(df)
    df = df.groupby("name").agg({"area": lambda x: list(x), "conf": lambda x: list(x)})
    out_path.mkdir(exist_ok=True, parents=True)
    cur_row = 0
    for row in tqdm(df.itertuples(), total=len(df)):
        row_name = row.Index
        areas = row.area
        max_area = max(areas)
        idx = areas.index(max_area)
        conf = row.conf[idx]
        cur_row += len(areas)
        # if max_area < 250_000 and max_area > 200_000:
        img_path = image_dir / row_name
        if max_area > 280_000 and conf < 0.7:
        # if max_area > 280_000 and max_area < 350_000:
            assert img_path.exists()
            shutil.copy(img_path, out_path / row_name)

        # if len(areas) > 1:
        #     multiface = out_path.parent / "multiface"
        #     multiface.mkdir(exist_ok=True, parents=True)
        #     shutil.copy(img_path, multiface / row_name)
            
    assert expected_row == cur_row

@app.command()
def no_detection(
    csv_path: Path = typer.Argument(..., help="csv path", file_okay=True, exists=True),
    image_dir: Path = typer.Argument(..., help="Image Dir", dir_okay=True, exists=True),
    out_path: Path = typer.Argument(..., help="out path"),
):
    df = pd.read_csv(csv_path)
    names = set(df["name"])
    files = set(map(lambda x: x.name, image_dir.glob("*.png")))
    no_det_files = files - names
    out_path.mkdir(exist_ok=True,parents=True)
    for file in tqdm(no_det_files):
        shutil.copy(image_dir / file, out_path / file)

@app.command()
def generate_filelist(
    csv_path: Path = typer.Argument(..., help="csv path", file_okay=True, exists=True),
    image_dir: Path = typer.Argument(..., help="Image Dir", dir_okay=True, exists=True),
    out_path: Path = typer.Argument(..., help="out path", file_okay=True),
):
    df = pd.read_csv(csv_path)
    names = set(df["name"])
    print(max(df["area"]))
    expected_row = len(df)
    df = df.groupby("name").agg({"area": lambda x: list(x), "conf": lambda x: list(x)})

    out_path.parent.mkdir(exist_ok=True, parents=True)

    cur_row = 0
    filter_files = []
    for row in tqdm(df.itertuples(), total=len(df)):
        row_name = row.Index
        areas = row.area
        max_area = max(areas)
        idx = areas.index(max_area)
        conf = row.conf[idx]
        cur_row += len(areas)
        # if max_area < 250_000 and max_area > 200_000:
        img_path = image_dir / row_name
        criteria_1 = (max_area > 280_000 and conf < 0.7)
        criteria_2 = (max_area < 280_000)
        if criteria_1 or criteria_2:
            filter_files.append(row_name)
    assert expected_row == cur_row
    # No detections
    files = set(map(lambda x: x.name, image_dir.glob("*.png")))
    no_det_files = files - names
    filter_files = set(filter_files)
    filter_files = filter_files.union(no_det_files)
    print(f"Total filtered files: {len(filter_files)}")
    with open(out_path, "w") as f:
        for file in filter_files:
            f.write(f"{file}\n")
            
if __name__ == "__main__":
    app()