import json
import shutil
from pathlib import Path

import pandas as pd
import typer
from natsort import natsorted
from tqdm import tqdm

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
        # if max_area > 280_000 and conf < 0.7:
        # if max_area > 100_000 and max_area > 80_000:
        if max_area > 100_000 and conf < 0.5:
            # if max_area > 280_000 and max_area < 350_000:
            if img_path.exists():
                shutil.copy(img_path, out_path / row_name)

        # if len(areas) > 1:
        #     multiface = out_path.parent / "multiface"
        #     multiface.mkdir(exist_ok=True, parents=True)
        #     shutil.copy(img_path, multiface / row_name)

    assert expected_row == cur_row


@app.command()
def area_deleted_to_filelist(
    csv_path: Path = typer.Argument(..., help="csv path", file_okay=True, exists=True),
    image_dir: Path = typer.Argument(..., help="Image Dir", dir_okay=True, exists=True),
    out_path: Path = typer.Argument(..., help="out path", file_okay=True),
):
    df = pd.read_csv(csv_path)
    expected_row = len(df)
    df = df.groupby("name").agg({"area": lambda x: list(x), "conf": lambda x: list(x)})
    out_path.parent.mkdir(exist_ok=True, parents=True)
    cur_row = 0
    f = open(out_path, "w")
    for row in tqdm(df.itertuples(), total=len(df)):
        row_name = row.Index
        areas = row.area
        max_area = max(areas)
        idx = areas.index(max_area)
        conf = row.conf[idx]
        cur_row += len(areas)
        # if max_area < 250_000 and max_area > 200_000:
        img_path = image_dir / row_name
        # if max_area > 280_000 and conf < 0.7:
        if max_area > 60_000 and max_area < 80_000:
            # if max_area > 100_000 and max_area > 80_000:
            if not img_path.exists():
                f.write(f"{row_name}\n")
    f.close()


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
    out_path.mkdir(exist_ok=True, parents=True)
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
        criteria_1 = max_area > 280_000 and conf < 0.7
        criteria_2 = max_area < 280_000
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


@app.command()
def merge_eg3d(
    image_dir: Path = typer.Argument(..., help="Image Dir", dir_okay=True, exists=True),
    ffhq_dataset_json_path: Path = typer.Argument(
        ..., help="csv path", file_okay=True, exists=True
    ),
    celeb_image_path: Path = typer.Argument(
        ..., help="out path", file_okay=True, exists=True
    ),
    celeb_json_path: Path = typer.Argument(
        ..., help="out path", file_okay=True, exists=True
    ),
    celeb_filter_path: Path = typer.Option(
        None, help="out path", file_okay=True, exists=True
    ),
    vfhq_image_path: Path = typer.Argument(
        ..., help="out path", file_okay=True, exists=True
    ),
    vfhq_json_path: Path = typer.Argument(
        ..., help="out path", file_okay=True, exists=True
    ),
    vfhq_filter_path: Path = typer.Option(
        None, help="out path", file_okay=True, exists=True
    ),
    start_fileidx: int = typer.Option(139914, help="start index"),
    start_folderidx: int = typer.Option(139, help="start index"),
):
    def read_labelobj_to_dict(path):
        with open(path, "r") as f:
            label_obj = json.load(f)
        d = {}
        for label in label_obj["labels"]:
            d[label[0]] = label[1]
        return d

    with open(ffhq_dataset_json_path, "r") as f:
        label_objs = json.load(f)
    celeb_lookup = read_labelobj_to_dict(celeb_json_path)
    vfhq_lookup = read_labelobj_to_dict(vfhq_json_path)

    celeb_filter = set()
    vfhq_filter = set()
    if celeb_filter_path is not None:
        with open(celeb_filter_path, "r") as f:
            celeb_filter = set(map(lambda x: x.strip(), f.readlines()))
    if vfhq_filter_path is not None:
        with open(vfhq_filter_path, "r") as f:
            vfhq_filter = set(map(lambda x: x.strip(), f.readlines()))

    print(f"celeb filter len: {len(celeb_filter)}")
    print(f"vfhq filter len: {len(vfhq_filter)}")
    celeb_image_paths = natsorted(list(celeb_image_path.glob("*.png")))
    cur_file_idx = start_fileidx
    lookup = {}
    for file in tqdm(celeb_image_paths, desc="copy celeb"):
        if file.name in celeb_filter:
            continue
        if cur_file_idx % 1000 == 0:
            start_folderidx += 1
        parent_name = str(start_folderidx).zfill(5)
        fname = str(cur_file_idx).zfill(8)
        fname = f"{parent_name}/img{fname}.png"
        dest = image_dir / fname
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(file, dest)
        label_objs["labels"].append([fname, celeb_lookup[file.name]])
        cur_file_idx += 1
        lookup["celeb/" + file.name] = fname

    vfhq_image_paths = natsorted(list(vfhq_image_path.glob("*.png")))
    for file in tqdm(vfhq_image_paths, desc="copy vfhq"):
        if file.name in vfhq_filter:
            continue
        if cur_file_idx % 1000 == 0:
            start_folderidx += 1
        parent_name = str(start_folderidx).zfill(5)
        fname = str(cur_file_idx).zfill(8)
        fname = f"{parent_name}/img{fname}.png"
        dest = image_dir / fname
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(file, dest)
        label_objs["labels"].append([fname, vfhq_lookup[file.name]])
        cur_file_idx += 1
        lookup["vfhq/" + file.name] = fname

    with open(image_dir / "dataset_merged.json", "w") as f:
        json.dump(label_objs, f, indent=2)
    with open(image_dir / "lookup.json", "w") as f:
        json.dump(lookup, f, indent=2)


if __name__ == "__main__":
    app()
