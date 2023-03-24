import json
import os
import re
import shutil
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional
import concurrent.futures

import matplotlib.pyplot as plt
from rich import print
import numpy as np
import pandas as pd
import typer
from cython_bbox import bbox_overlaps as bbox_ious
from natsort import natsorted
from pydantic import BaseModel
from tqdm import tqdm

from .helper import *
from .metadata import *

app = typer.Typer()


@app.command()
def ffhq_poseify(
    input_csv: Path = typer.Argument(
        ..., help="input path", exists=True, file_okay=True
    ),
    plot_dist: bool = typer.Option(True, help="plot pose dist"),
    output_path: Path = typer.Option(".", help="output path"),
    prefix: str = typer.Option("", help="prefix to name"),
):
    df = pd.read_csv(input_csv)
    output_path.mkdir(parents=True, exist_ok=True)
    yaws = df["head_yaw"].to_list()
    pitchs = df["head_pitch"].to_list()
    if plot_dist:
        plt.hist(yaws, bins=range(-90, 90, 10), alpha=0.5, label="Yaw")
        plt.hist(pitchs, bins=range(-90, 90, 10), alpha=0.5, label="Pitch")
        plt.title("FFHQ's Pitch, Yaw distribution")
        plt.savefig((output_path / "pitch_yaw.png").as_posix())
    labels = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_name = prefix + str(row.image_number).zfill(5) + ".png"
        label = "frontal"
        if row.head_yaw > 45:
            label = "profile_left"
        elif row.head_yaw < -45:
            label = "profile_right"
        elif row.head_pitch > 30:
            label = "profile_up"
        elif row.head_pitch < -30:
            label = "profile_down"
        pair = [img_name, label]
        labels.append(pair)
    out = {"labels": labels}
    with open(output_path / "datasets.json", "w") as f:
        json.dump(out, f)


@app.command()
def selfgen_poseify(
    input_basepath: Path = typer.Argument(
        ..., help="input path", exists=True, dir_okay=True
    ),
    output_path: Path = typer.Option(".", help="output path"),
    prefix: str = typer.Option("", help="prefix to name"),
):
    txt_files = input_basepath.rglob("*/info.txt")
    labels = []
    for i, txt_file in enumerate(txt_files):
        if i % 2000 == 0:
            print(f"Process {i} files...")
        rela_path = txt_file.relative_to(input_basepath)
        with open(txt_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2
        line = lines[-1].strip().split()
        filename = line[0]
        filepath = rela_path.parent / f"{filename}.png"
        assert (input_basepath / filepath).exists(), f"{filepath} not exists."
        pitch = float(line[-3])
        yaw = float(line[-2])
        label = "frontal"
        if yaw > 45:
            label = "profile_left"
        elif yaw < -45:
            label = "profile_right"
        elif pitch > 30:
            label = "profile_up"
        elif pitch < -30:
            label = "profile_down"
        pair = [filepath.as_posix(), label]
        labels.append(pair)

    out = {"labels": labels}
    with open(output_path / "datasets.json", "w") as f:
        json.dump(out, f)

    pass


@app.command()
def merge_jsons(
    input_basepath: Path = typer.Argument(
        ..., help="input path", exists=True, dir_okay=True
    ),
    output_path: Path = typer.Argument(..., help="output path"),
    plot_dist: bool = typer.Option(True, help="plot pose dist"),
):
    jsons = input_basepath.rglob("*/datasets.json")
    labels = []
    for json_p in jsons:
        with open(json_p, "r") as f:
            tmp = json.load(f)
        labels.extend(tmp["labels"])
    d = defaultdict(int)
    for label in labels:
        if "test" in label[0] or "train" in label[0]:
            label[0] = "pairs_iqa60+/" + label[0]
        d[label[1]] += 1

    if plot_dist:
        output_path.mkdir(parents=True, exist_ok=True)
        plt.bar(list(d.keys()), list(d.values()), color="maroon")
        plt.xlabel("Pose")
        plt.ylabel("Count")
        plt.title("Dataset label distribution")
        plt.savefig((output_path / "pitch_yaw.png").as_posix())

    out = {"labels": labels}
    with open(output_path / "datasets.json", "w") as f:
        json.dump(out, f)


@app.command()
def vfhq_directmhp_merge(
    directmhp_path: Path = typer.Argument(
        ..., help="directmhp path", exists=True, dir_okay=True
    ),
    gt_path: Path = typer.Argument(..., help="gt path", exists=True, dir_okay=True),
    out_path: Path = typer.Argument(..., help="out path"),
    iou_thresh: float = typer.Option(0.8, help="iou thresh"),
):
    gt_csvs = natsorted(gt_path.glob("*.csv"))
    out_path.mkdir(parents=True, exist_ok=True)
    for gt_csv in tqdm(gt_csvs):
        videoid = gt_csv.stem
        df_ = pd.read_csv(gt_csv)
        ref = {}
        # group dataframe by column name "frameid", all other columns should be aggregated
        df = df_.groupby("frameid", as_index=False).agg(
            {
                "idx": lambda x: list(x),
                "x1": lambda x: list(x),
                "y1": lambda x: list(x),
                "x2": lambda x: list(x),
                "y2": lambda x: list(x),
            }
        )
        records = df.to_dict("records")
        for row in records:
            frameid = row["frameid"]
            txtpath = directmhp_path / videoid / f"{str(frameid).zfill(8)}.txt"
            if not txtpath.exists():
                continue
            with open(txtpath, "r") as f:
                lines = f.readlines()
            if len(lines) == 0:
                continue
            lines = [line.strip().split() for line in lines]
            yprltrbs = []
            for line in lines:
                yprltrbs.append(list(map(float, line)))
            pred_yprltrbs = np.array(yprltrbs)
            if len(line) == 1:
                ref[f"{frameid}_{ids[0]}"] = pred_yprltrbs[0, :3]
            else:
                ids = row["idx"]
                x1s = row["x1"]
                y1s = row["y1"]
                x2s = row["x2"]
                y2s = row["y2"]
                ltrbs = np.array(list(zip(x1s, y1s, x2s, y2s)))
                if (ltrbs[:, 2] < ltrbs[:, 0]).all():
                    ltrbs[:, 2] += ltrbs[:, 0]
                    ltrbs[:, 3] += ltrbs[:, 1]

                ious_ = ious(ltrbs, pred_yprltrbs[:, 3:])
                max_ids, max_ious = np.argmax(ious_, axis=-1), np.max(ious_, axis=-1)
                for i, (max_idx, max_iou) in enumerate(zip(max_ids, max_ious)):
                    if max_iou > iou_thresh:
                        ref[f"{frameid}_{ids[i]}"] = pred_yprltrbs[max_idx, :3]
        df_["mhp_yaw"] = np.nan
        df_["mhp_pitch"] = np.nan
        df_["mhp_roll"] = np.nan

        for idx, row in df_.iterrows():
            frameid = row["frameid"]
            fidx = row["idx"]
            key = f"{frameid}_{fidx}"
            if key in ref:
                df_.loc[idx, "mhp_yaw"] = ref[key][0]
                df_.loc[idx, "mhp_pitch"] = ref[key][1]
                df_.loc[idx, "mhp_roll"] = ref[key][2]
        outcsvpath = out_path / gt_csv.name
        df_.to_csv(
            outcsvpath.as_posix(),
            index=False,
            columns=[
                "frameid",
                "idx",
                "x1",
                "y1",
                "x2",
                "y2",
                "iqa",
                "synergy_yaw",
                "synergy_pitch",
                "synergy_roll",
                "poseanh_yaw",
                "poseanh_pitch",
                "poseanh_roll",
                "mhp_yaw",
                "mhp_pitch",
                "mhp_roll",
                "lmks5pts",
                "lmks68pts",
            ],
        )


@app.command()
def vfhq_posemerge(
    synergy_path: Path = typer.Argument(
        ..., help="synergy path", exists=True, dir_okay=True
    ),
    poseanh_path: Path = typer.Argument(
        ..., help="poseanh path", exists=True, dir_okay=True
    ),
    iqa_path: Path = typer.Argument(..., help="iqa path", exists=True, dir_okay=True),
    gt_path: Path = typer.Argument(..., help="gt path", exists=True, dir_okay=True),
    output_path: Path = typer.Argument(..., help="output path"),
):
    synergytxts = synergy_path.glob("*.txt")
    poseanhtxts = poseanh_path.glob("*.txt")
    iqatxts = iqa_path.glob("*.txt")
    gttxts = gt_path.glob("*.txt")

    synergynames = set([x.stem for x in synergytxts])
    poseanhnames = set([x.stem for x in poseanhtxts])
    iqanames = set([x.stem for x in iqatxts])
    gtnames = set([x.stem for x in gttxts])

    overlapnames = synergynames & poseanhnames & gtnames & iqanames
    missingnames = gtnames - overlapnames
    print(f"Total clips matched: {len(overlapnames)}")
    print(f"Total clips unmatched comparing to GT: {len(missingnames)}")

    missing_outpath = output_path / "missing.txt"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(missing_outpath.as_posix(), "w") as f:
        for missingname in missingnames:
            f.write(missingname + "\n")

    for overlapname in tqdm(overlapnames):
        synergytxtpath = synergy_path / (overlapname + ".txt")
        poseanhtxtpath = poseanh_path / (overlapname + ".txt")
        gttxtpath = gt_path / (overlapname + ".txt")
        iqatxtpath = iqa_path / (overlapname + ".txt")

        df = mergetxt(gttxtpath, synergytxtpath, poseanhtxtpath, iqatxtpath)
        outputdfpath = output_path / (overlapname + ".csv")
        df.to_csv(
            outputdfpath,
            index=False,
            columns=[
                "frameid",
                "idx",
                "x1",
                "y1",
                "x2",
                "y2",
                "synergy_yaw",
                "synergy_pitch",
                "synergy_roll",
                "poseanh_yaw",
                "poseanh_pitch",
                "poseanh_roll",
                "lmks5pts",
                "lmks68pts",
            ],
        )


@app.command()
def vfhq_posemerge_multithread(
    synergy_path: Path = typer.Argument(
        ..., help="synergy path", exists=True, dir_okay=True
    ),
    poseanh_path: Path = typer.Argument(
        ..., help="poseanh path", exists=True, dir_okay=True
    ),
    iqa_path: Path = typer.Argument(..., help="iqa path", exists=True, dir_okay=True),
    gt_path: Path = typer.Argument(..., help="gt path", exists=True, dir_okay=True),
    output_path: Path = typer.Argument(..., help="output path"),
    workers: int = typer.Option(8, help="nworkers"),
):
    synergytxts = synergy_path.glob("*.txt")
    poseanhtxts = poseanh_path.glob("*.txt")
    iqatxts = iqa_path.glob("*.txt")
    gttxts = gt_path.glob("*.txt")

    synergynames = set([x.stem for x in synergytxts])
    poseanhnames = set([x.stem for x in poseanhtxts])
    iqanames = set([x.stem for x in iqatxts])
    gtnames = set([x.stem for x in gttxts])

    overlapnames = synergynames & poseanhnames & gtnames & iqanames
    missingnames = gtnames - overlapnames
    print(f"Total clips matched: {len(overlapnames)}")
    print(f"Total clips unmatched comparing to GT: {len(missingnames)}")

    missing_outpath = output_path / "missing.txt"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(missing_outpath.as_posix(), "w") as f:
        for missingname in missingnames:
            f.write(missingname + "\n")

    def func(overlapname):
        synergytxtpath = synergy_path / (overlapname + ".txt")
        poseanhtxtpath = poseanh_path / (overlapname + ".txt")
        gttxtpath = gt_path / (overlapname + ".txt")
        iqatxtpath = iqa_path / (overlapname + ".txt")
        outputdfpath = output_path / (overlapname + ".csv")

        df = mergetxt(gttxtpath, synergytxtpath, poseanhtxtpath, iqatxtpath)
        df.to_csv(
            outputdfpath,
            index=False,
            columns=[
                "frameid",
                "idx",
                "x1",
                "y1",
                "x2",
                "y2",
                "iqa",
                "synergy_yaw",
                "synergy_pitch",
                "synergy_roll",
                "poseanh_yaw",
                "poseanh_pitch",
                "poseanh_roll",
                "lmks5pts",
                "lmks68pts",
            ],
        )

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    results = []
    for result in tqdm(
        pool.map(func, natsorted(overlapnames)), total=len(overlapnames)
    ):
        results.append(result)


@app.command()
def vfhq_combine_multiid_into_one(
    csv_basepath: Path = typer.Argument(
        ..., help="csvpath", dir_okay=True, exists=True
    ),
    out_basepath: Path = typer.Argument(..., help="outputpath"),
):
    csv_paths = list(csv_basepath.glob("*.csv"))
    csv_names = [x.stem for x in csv_paths]
    video_ids = list([x.split("+")[1] for x in csv_names])
    d = defaultdict(list)
    assert len(video_ids) == len(csv_paths)
    out_basepath.mkdir(parents=True, exist_ok=True)
    for i, (video_id, csv_path) in enumerate(zip(video_ids, csv_paths)):
        d[video_id].append(csv_path)

    for video_id, csv_paths in tqdm(d.items(), total=len(d)):
        dfs = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path.as_posix())
            dfs.append(df)
        result = pd.concat(dfs, ignore_index=True)
        outcsvpath = out_basepath / (video_id + ".csv")
        result.to_csv(outcsvpath.as_posix(), index=False)


@app.command()
def binning(
    csv_path: Path = typer.Argument(..., help="csvpath", exists=True, dir_okay=True),
    output_path: Path = typer.Argument(..., help="outpath"),
):
    csv_paths = list(natsorted(csv_path.glob("*.csv")))
    output_path.mkdir(exist_ok=True, parents=True)
    pbar = tqdm(csv_paths)
    hardcounter = defaultdict(int)
    softcounter = defaultdict(int)
    for csv_path in pbar:
        pbardesc = f"{csv_path.stem}"
        if "confused" in hardcounter:
            pbardesc += f"|H{hardcounter['confused']}"
        if "confused" in softcounter:
            pbardesc += f"|E{softcounter['confused']}"
        pbar.set_description(pbardesc)
        df = pd.read_csv(csv_path.as_posix())
        records = df.to_dict("records")
        for row_idx, row in enumerate(tqdm(records)):
            synergy_yaw = row["synergy_yaw"]
            synergy_pitch = row["synergy_pitch"]
            synergy_roll = row["synergy_roll"]
            poseanh_yaw = row["poseanh_yaw"]
            poseanh_pitch = row["poseanh_pitch"]
            poseanh_roll = row["poseanh_roll"]
            mhp_yaw = row["mhp_yaw"]
            mhp_pitch = row["mhp_pitch"]
            mhp_roll = row["mhp_roll"]

            synergy_bin = bin_a_pose(synergy_yaw, synergy_pitch, synergy_roll)
            poseanh_bin = bin_a_pose(poseanh_yaw, poseanh_pitch, poseanh_roll)
            mhp_bin = bin_a_pose(mhp_yaw, mhp_pitch, mhp_roll)
            df.loc[row_idx, "synergy_bin"] = synergy_bin
            df.loc[row_idx, "poseanh_bin"] = poseanh_bin
            df.loc[row_idx, "mhp_bin"] = mhp_bin

            if synergy_bin == poseanh_bin == mhp_bin:
                hard_bin = synergy_bin
            else:
                hard_bin = "confused"

            is_profile = all(
                [
                    x.split("_")[0] == "profile"
                    for x in [synergy_bin, poseanh_bin, mhp_bin]
                ]
            )
            is_frontal = all(
                [x == "frontal" for x in [synergy_bin, poseanh_bin, mhp_bin]]
            )
            if is_frontal:
                soft_bin = "frontal"
            elif is_profile:
                is_horizontal = all(
                    [
                        x.split("_")[1] in ["left", "right", "extreme"]
                        for x in [synergy_bin, poseanh_bin, mhp_bin]
                    ]
                )
                is_vertical = all(
                    [
                        x.split("_")[1] in ["up", "down", "extreme"]
                        for x in [synergy_bin, poseanh_bin, mhp_bin]
                    ]
                )
                if is_horizontal:
                    soft_bin = "profile_horizontal"
                elif is_vertical:
                    soft_bin = "profile_vertical"
                else:
                    soft_bin = "confused"
            else:
                soft_bin = "confused"

            df.loc[row_idx, "hardbin"] = hard_bin
            df.loc[row_idx, "softbin"] = soft_bin
            hardcounter[hard_bin] += 1
            softcounter[soft_bin] += 1
        outputcsvpath = output_path / csv_path.name
        df.to_csv(outputcsvpath.as_posix(), index=False)

    print("Hard: {hardcounter}")
    print("Soft: {softcounter}")

if __name__ == "__main__":
    app()
