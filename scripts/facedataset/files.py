from pathlib import Path
from matplotlib import pyplot as plt
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
def filter(
    brightness_csv_path: Path = typer.Argument(
        ..., help="csv path", file_okay=True, exists=True
    ),
    vfhq_filter_path: Path = typer.Argument(
        ..., help="csv path", file_okay=True, exists=True
    ),
    celeb_filter_path: Path = typer.Argument(
        ..., help="csv path", file_okay=True, exists=True
    ),
    vfhq_image_dir: Path = typer.Argument(
        ..., help="Image Dir", dir_okay=True, exists=True
    ),
    celeb_image_dir: Path = typer.Argument(
        ..., help="Image Dir", dir_okay=True, exists=True
    ),
    out_path: Path = typer.Argument(..., help="out path"),
):
    df = pd.read_csv(brightness_csv_path)
    with open(vfhq_filter_path, "r") as f:
        vfhqfiles = f.readlines()
    vfhqfiles = set(map(lambda x: x.strip(), vfhqfiles))
    with open(celeb_filter_path, "r") as f:
        celebfiles = f.readlines()
    celebfiles = set(map(lambda x: x.strip(), celebfiles))
    for row in df.itertuples():
        name = row.path
        if row.brightness <= 40:
            if "celeb" in name:
                celebfiles.add(Path(name).name)
            elif "vfhq" in name:
                vfhqfiles.add(Path(name).name)
    raw_vfhq_files = set(map(lambda x: x.name, vfhq_image_dir.glob("*.png")))
    raw_celeb_files = set(map(lambda x: x.name, celeb_image_dir.glob("*.png")))

    assert celebfiles.issubset(raw_celeb_files)
    assert vfhqfiles.issubset(raw_vfhq_files)

    final_vfhq_files = raw_vfhq_files - vfhqfiles
    final_celeb_files = raw_celeb_files - celebfiles
    vfhq_out_path = out_path / "vfhq"
    celeb_out_path = out_path / "celeb"
    vfhq_out_path.mkdir(exist_ok=True, parents=True)
    celeb_out_path.mkdir(exist_ok=True, parents=True)

    for file in final_vfhq_files:
        shutil.copy(vfhq_image_dir / file, vfhq_out_path / file)
    for file in final_celeb_files:
        shutil.copy(celeb_image_dir / file, celeb_out_path / file)

@app.command()
def count_pose_celebvhq(
    img_dir: Path = typer.Argument(..., help="img dir", dir_okay=True, exists=True)
):
    files = list(img_dir.glob("*.png"))
    extreme_yaw = []
    extreme_pitch = []
    for file in files:
        name = file.stem
        splits = name.split("_")
        y,p = tuple(map(int, splits[2:]))
        print(y, p)
        if abs(y) > 40:
            extreme_yaw.append(y)
        if abs(p) > 30:
            extreme_pitch.append(p)
    print(f"#yaw={len(extreme_yaw)}")
    print(f"#pitch={len(extreme_pitch)}")
    plt.figure(1)
    plt.hist(extreme_pitch, bins = range(-100,100,10)) 
    plt.title("Extreme pitch: CelebVHQ") 
    plt.savefig("/lustre/scratch/client/vinai/users/trungdt21/workspace/common/prata/playground/facedataset/celebvhq/hist_pitch.png")
    plt.figure(2)
    plt.hist(extreme_yaw, bins = range(-100,100,10)) 
    plt.title("Extreme yaw: CelebVHQ") 
    plt.savefig("/lustre/scratch/client/vinai/users/trungdt21/workspace/common/prata/playground/facedataset/celebvhq/hist_yaw.png")

            
@app.command()
def count_pose_vfhq(
    img_dir: Path = typer.Argument(..., help="img dir", dir_okay=True, exists=True)
):
    files = list(img_dir.glob("*.png"))
    extreme_yaw = []
    extreme_pitch = []
    for file in files:
        name = file.stem
        splits = name.split("_")
        yprstart_idx = -1
        for i, split in enumerate(splits):
            if "ypr" in split:
                yprstart_idx = i
                break
        y = int(splits[i].replace("ypr", ""))
        p = int(splits[i+1])
            
        print(y, p)
        if abs(y) > 40:
            extreme_yaw.append(y)
        if abs(p) > 30:
            extreme_pitch.append(p)
    print(f"#yaw={len(extreme_yaw)}")
    print(f"#pitch={len(extreme_pitch)}")
    plt.figure(1)
    plt.hist(extreme_pitch, bins = range(-100,100,10)) 
    plt.title("Extreme pitch: VFHQ") 
    plt.savefig("/lustre/scratch/client/vinai/users/trungdt21/workspace/common/prata/playground/facedataset/vfhq/hist_pitch.png")
    plt.figure(2)
    plt.hist(extreme_yaw, bins = range(-100,100,10)) 
    plt.title("Extreme yaw: VFHQ") 
    plt.savefig("/lustre/scratch/client/vinai/users/trungdt21/workspace/common/prata/playground/facedataset/vfhq/hist_yaw.png")

if __name__ == "__main__":
    app()
