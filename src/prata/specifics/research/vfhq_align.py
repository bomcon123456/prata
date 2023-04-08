from pathlib import Path
import math
import json
import typer
from typing import List
import concurrent.futures

import numpy as np
import cv2
from tqdm.rich import tqdm
import pandas as pd

from .face_align import image_align_5
from .zipdataset import ImageZipDataset


def get_pose(row: tuple):
    if math.isnan(row.mhp_yaw):
        y, p, r = row.synergy_yaw, row.synergy_pitch, row.synergy_roll
    else:
        y, p, r = row.mhp_yaw, row.mhp_pitch, row.mhp_roll
    return int(y), int(p), int(r)


def aligner(
    zippath: Path = typer.Argument(..., help="zippath"),
    csv_dir: Path = typer.Argument(..., help="csv_dir"),
    output_basepath: Path = typer.Argument(..., help="Path to data.yaml"),
    workers: int = typer.Option(16, help="num workers"),
):
    def func(zip_path: Path):
        id_name = zippath.stem
        csv_path = csv_dir / f"{id_name}.csv"
        curid_outpath = output_basepath / id_name
        curid_outpath.mkdir(exist_ok=True, parents=True)

        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        parent_path = output_basepath / zippath.stem
        parent_path.mkdir(exist_ok=True, parents=True)
        dataset_wrapper = ImageZipDataset(zippath)
        with dataset_wrapper.dataset() as dataset:
            dataset_iter = iter(dataset)
            for index in tqdm(range(len(dataset))):
                (single_path, img) = next(dataset_iter)
                single_path = Path(single_path)
                frameid = single_path.stem.lstrip("0")
                rows = df[df["frameid"] == frameid]
                assert len(rows) > 0
                for row in rows.itertuples():
                    yaw, pitch, roll = get_pose(row)
                    softbin = row.softbin
                    lmk5pts = np.array(json.loads(row["lmks5pts"])).reshape(-1, 2)
                    img = image_align_5(img, lmk5pts, lmk5pts)
                    img_outpath = (
                        curid_outpath
                        / f"{row.Index}_ypr{yaw}_{pitch}_{roll}_{softbin}.png"
                    )
                    cv2.imwrite(img_outpath, img)
                    df.loc[row.Index, "aligned_path"] = img_outpath.relative_to(output_basepath).as_posix()

        dataset_wrapper.zipfile.close()
        df.to_csv((output_basepath / csv_path.name).as_posix(), index=False)

    paths = []
    if zippath.is_file():
        if zippath.suffix.lower == ".zip":
            paths = [zippath]
        elif zippath.suffix == ".txt":
            with open(zippath, "r") as f:
                lines = f.readlines()
                paths = list(map(lambda x: Path(x.strip()), lines))
    elif zippath.is_dir():
        paths = zippath.glob("*.zip")

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    results = []
    for result in tqdm(pool.map(func, paths), total=len(paths)):
        results.append(result)
