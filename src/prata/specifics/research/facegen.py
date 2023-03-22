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
    iou_thresh: float = typer.Option(0.8, help="iou thresh"),
):
    video_ids = os.listdir(directmhp_path)
    video_paths = list([directmhp_path / video_id for video_id in video_ids])

    gt_files = list(gt_path.rglob("*.txt"))
    clipid_to_txts = defaultdict(list)
    pbar = tqdm(gt_files)
    pbar.set_description("Processing gt files")
    for gt_file in pbar:
        name = gt_file.stem
        clip_id = name.split("+")[1]
        clipid_to_txts[clip_id].append(gt_file)

    # clipid_to_fidbox = dict()
    for videopath in pbar:
        pbar.set_description(f"Process video: {videopath.stem}")
        clipid = videopath.stem
        gttxts = clipid_to_txts[clipid]
        frameid_to_bbox = defaultdict(list)
        for gttxt in gttxts:
            with open(gttxt, "r") as f:
                lines = f.readlines()[7:-1]
            for line in lines:
                line = line.strip()
                line = re.sub(r"\s+", " ", line)
                facepred: FacePrediction = FacePrediction.parse_vfhq_annotation(
                    line, clipid
                )
                frameid_to_bbox[facepred.frame_id].append(facepred)
        predtxts = natsorted(videopath.rglob("*.txt"))
        for predtxt in predtxts:
            listbboxes = SimpleBoundingBox.parse_from_file(predtxt)
            frameid = predtxt.stem
            gtboxes: List[BoundingBox] = frameid_to_bbox[frameid]
            targetious = [np.array(x.ltrb_gt) for x in gtboxes]
            srcious = [np.array(x.ltrb) for x in listbboxes]
            ious_ = ious(srcious, targetious)
            max_ids, max_ious = np.argmax(ious_, axis=1), np.max(ious, axis=1)
            numuniques = np.unique(max_ids).shape[0]
            assert numuniques == max_ids.shape[0], f"There are multiple boxes overlap the same gtbbox {max_ids}, {max_ious}"
            for max_id, max_iou in zip(max_ids, max_ious):
                if max_iou > iou_thresh:
                    listbboxes[max_id].gtbox = gtboxes[max_id]


@app.command()
def vfhq_posemerge(
    synergy_path: Path = typer.Argument(
        ..., help="synergy path", exists=True, dir_okay=True
    ),
    poseanh_path: Path = typer.Argument(
        ..., help="poseanh path", exists=True, dir_okay=True
    ),
    iqa_path: Path = typer.Argument(
        ..., help="iqa path", exists=True, dir_okay=True
    ),
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
    iqa_path: Path = typer.Argument(
        ..., help="iqa path", exists=True, dir_okay=True
    ),
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
    for result in tqdm(pool.map(func, natsorted(overlapnames)), total=len(overlapnames)):
        results.append(result)

if __name__ == "__main__":
    app()
