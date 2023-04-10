import concurrent.futures
import json
import math
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from .face_align import image_align_5
from .zipdataset import ImageZipDataset


def get_pose(row: tuple):
    if math.isnan(row.mhp_yaw):
        y, p, r = row.synergy_yaw, row.synergy_pitch, row.synergy_roll
    else:
        y, p, r = row.mhp_yaw, row.mhp_pitch, row.mhp_roll
    return int(y), int(p), int(r)


def get_pose_dict(row: Dict):
    if math.isnan(row["mhp_yaw"]):
        y, p, r = row["synergy_yaw"], row["synergy_pitch"], row["synergy_roll"]
    else:
        y, p, r = row["mhp_yaw"], row["mhp_pitch"], row["mhp_roll"]
    return int(y), int(p), int(r)


def aligner_zip(
    zippath: Path = typer.Argument(..., help="zippath"),
    csv_dir: Path = typer.Argument(..., help="csv_dir"),
    output_basepath: Path = typer.Argument(..., help="Path to data.yaml"),
    workers: int = typer.Option(16, help="num workers"),
):
    def func(zip_path: Path):
        id_name = zip_path.stem
        csv_path = csv_dir / f"{id_name}.csv"
        curid_outpath = output_basepath / id_name
        curid_outpath.mkdir(exist_ok=True, parents=True)

        assert csv_path.exists(), f"csv path: {csv_path} is not existed!"
        df = pd.read_csv(csv_path)
        parent_path = output_basepath / zippath.stem
        parent_path.mkdir(exist_ok=True, parents=True)
        dataset_wrapper = ImageZipDataset(zip_path)
        with dataset_wrapper.dataset() as dataset:
            dataset_iter = iter(dataset)
            for index in tqdm(range(len(dataset))):
                (single_path, img) = next(dataset_iter)
                single_path = Path(single_path)
                frameid = single_path.stem.lstrip("0")
                rows = df[df["frameid"] == int(frameid)]
                assert len(rows) > 0, f"Row is empty for {csv_path}, frameid={frameid}"
                for row in rows.itertuples():
                    yaw, pitch, roll = get_pose(row)
                    softbin = row.softbin
                    lmk5pts = np.array(json.loads(row.lmks5pts)).reshape(-1, 2)
                    img, _, _ = image_align_5(img, lmk5pts, lmk5pts)
                    img_outpath = (
                        curid_outpath
                        / f"{row.Index}_ypr{yaw}_{pitch}_{roll}_{softbin}.png"
                    )
                    cv2.imwrite(img_outpath.as_posix(), img)
                    df.loc[row.Index, "aligned_path"] = img_outpath.relative_to(
                        output_basepath
                    ).as_posix()

        dataset_wrapper.zipfile.close()
        csvoutdir = output_basepath / "csvs"
        csvoutdir.mkdir(parents=True, exist_ok=True)
        df.to_csv((csvoutdir / csv_path.name).as_posix(), index=False)

    paths = []
    if zippath.is_file():
        if zippath.suffix.lower == ".zip":
            paths = [zippath]
        elif zippath.suffix == ".txt":
            with open(zippath, "r") as f:
                lines = f.readlines()
                paths = list(map(lambda x: Path(x.strip()), lines))
    elif zippath.is_dir():
        paths = list(zippath.glob("*.zip"))

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    results = []
    for result in tqdm(pool.map(func, paths), total=len(paths)):
        results.append(result)


def aligner(
    imagebasepath: Path = typer.Argument(..., help="imgbasepath"),
    csv_dir: Path = typer.Argument(..., help="csv_dir"),
    output_basepath: Path = typer.Argument(..., help="Path to data.yaml"),
    workers: int = typer.Option(16, help="num workers"),
):
    def func(image_path: Path):
        id_name = image_path.stem
        frameid_to_path = {x.stem: x for x in image_path.rglob("*.png")}
        csv_path = csv_dir / f"{id_name}.csv"
        errorfullpath = errordir / f"{id_name}.txt"
        fout = None

        curid_outpath = output_basepath / id_name
        curid_outpath.mkdir(exist_ok=True, parents=True)

        assert csv_path.exists(), f"csv path: {csv_path} is not existed!"
        try:
            df = pd.read_csv(csv_path)
            csv_dict = df.to_dict("records")
            for row_idx, row in enumerate(tqdm(csv_dict)):
                frameid = str(row["frameid"]).zfill(8)
                img_path = frameid_to_path[frameid]
                img = cv2.imread(img_path.as_posix())
                yaw, pitch, roll = get_pose_dict(row)
                softbin = row["softbin"]
                lmkstr = row["lmks5pts"]
                if not isinstance(lmkstr, str):
                    if fout is None:
                        fout = open(errorfullpath, "w")
                    fout.write(f"Row idx: {row_idx}, lmk={lmkstr} is not an array\n")
                    continue
                lmk5pts = np.array(json.loads(row["lmks5pts"])).reshape(-1, 2)
                img, _, _ = image_align_5(img, lmk5pts, lmk5pts)
                img_outpath = (
                    curid_outpath / f"{row_idx}_ypr{yaw}_{pitch}_{roll}_{softbin}.png"
                )
                cv2.imwrite(img_outpath.as_posix(), img)
                df.loc[row_idx, "aligned_path"] = img_outpath.relative_to(
                    output_basepath
                ).as_posix()

            df.to_csv((csvoutdir / csv_path.name).as_posix(), index=False)
        except Exception as e:
            if fout is None:
                fout = open(errorfullpath, "w")
            fout.write(f"Error: {e}\n")

    paths = []
    if imagebasepath.is_file():
        if imagebasepath.suffix == ".txt":
            with open(imagebasepath, "r") as f:
                lines = f.readlines()
                paths = list(map(lambda x: Path(x.strip()), lines))
    elif imagebasepath.is_dir():
        paths = [imagebasepath]

    csvoutdir = output_basepath / "csvs"
    csvoutdir.mkdir(parents=True, exist_ok=True)
    errordir = output_basepath / "errors"
    errordir.mkdir(parents=True, exist_ok=True)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    results = []
    for result in tqdm(pool.map(func, paths), total=len(paths)):
        results.append(result)
