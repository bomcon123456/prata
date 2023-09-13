import typer
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import shutil
import numpy as np
import json

app = typer.Typer()


@app.command()
def main(
    csv_dir: Path = typer.Argument(
        ..., help="Base csv folder", dir_okay=True, exists=True
    ),
    aligned_dir: Path = typer.Argument(
        ..., help="Base aligned folder", dir_okay=True, exists=True
    ),
    outdir: Path = typer.Argument(..., help="Output dir"),
    image_per_idbin: int = typer.Option(10, help="How much sample per id_bin"),
):
    pose_dirs = set(os.listdir(csv_dir))
    if "frontal" in pose_dirs:
        pose_dirs.remove("frontal")
    outdir.mkdir(exist_ok=True, parents=True)
    outimgdir = outdir / "images"
    outmetafile = outdir / "metadata.json"
    outimgdir.mkdir(exist_ok=True, parents=True)
    pose_dirs = sorted(pose_dirs)
    info = {}
    image_counter = 0
    for pose_dir in tqdm(pose_dirs, desc="Pose folder"):
        pose_dir = csv_dir / pose_dir
        csv_files = list(pose_dir.rglob("*.csv"))
        for csv_file in tqdm(csv_files, desc="Parse csv"):
            df = pd.read_csv(csv_file)
            if len(df) > image_per_idbin:
                df.sort_values(by=["iqa"], inplace=True)
                df = df[:image_per_idbin]
            for row in df.itertuples():
                frame_idx, yaw, pitch, roll = (
                    row.frameid,
                    row.mhp_yaw,
                    row.mhp_pitch,
                    row.mhp_roll,
                )
                if np.isnan(yaw):
                    yaw, pitch, roll = (
                        row.synergy_yaw,
                        row.synergy_pitch,
                        row.synergy_roll,
                    )
                image_name = f"{image_counter}_yp_{int(yaw)}_{int(pitch)}.png"
                original_path = (
                    aligned_dir / csv_file.stem / f"{str(frame_idx).zfill(8)}_0.png"
                )
                if not original_path.exists():
                    print(f"{original_path} not exist")
                else:
                    shutil.copy2(original_path, outimgdir / image_name)
                    image_counter += 1
                    info[image_name] = {
                        "original_path": original_path.as_posix(),
                        "original_csv": csv_file.as_posix()
                    }
    with open(outmetafile, "w") as f:
        json.dump(info, f)



if __name__ == "__main__":
    app()
