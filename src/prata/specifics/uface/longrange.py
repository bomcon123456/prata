import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shortuuid
import typer
from natsort import natsorted
from tqdm import tqdm

from .face_aligncrop import norm_crop


def unify_label_for_track_id(df):
    groupdbytrackid = (
        df.groupby("track_id")["label"].apply(list).reset_index(name="label")
    )
    for row in groupdbytrackid.itertuples():
        tmp = filter(lambda x: x != "FACE_NOT_FOUND", row.label)
        dict_counter = Counter(tmp)
        if "unknown" in dict_counter:
            dict_counter["unknown"] //= 2

        k = dict_counter.most_common(1)
        if len(k) == 0:
            final_label = "unknown"
        else:
            final_label = k[0][0]
        df.loc[df["track_id"] == row.track_id, "label"] = final_label
    return df


app = typer.Typer()


def f1():
    basedf_path = Path(
        "/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/split_probe_gallery.csv"
    )
    posedf_path = Path(
        "/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/s103_out/csvs/s103_pose.csv"
    )
    magdf_path = Path(
        "/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/s103_out/csvs/s103_mag.csv"
    )
    maskdf_path = Path(
        "/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/s103_out/csvs/s103_mask.csv"
    )

    df = pd.read_csv(basedf_path)
    del df["Unnamed: 0"]
    del df["type"]
    del df["verify_type"]
    del df["masked"]
    posedf = pd.read_csv(posedf_path)
    magdf = pd.read_csv(magdf_path)
    maskdf = pd.read_csv(maskdf_path)

    for row in tqdm(df.itertuples(), total=len(df)):
        i = row.Index
        path = row.cropped_img_path.replace("cropped/", "")
        pose = posedf[posedf["fname"] == path]
        mag = magdf[magdf["fname"] == path]
        mask = maskdf[maskdf["fname"] == path]
        assert len(pose) == len(mag) == len(mask) == 1
        df.at[i, "is_mask"] = not bool(mask.iloc[0]["not_mask"])
        df.at[i, "pitch"] = float(pose.iloc[0]["pitch"])
        df.at[i, "yaw"] = float(pose.iloc[0]["yaw"])
        df.at[i, "roll"] = float(pose.iloc[0]["roll"])
        df.at[i, "mag"] = float(mag.iloc[0]["mag"])
    df.to_csv("out.csv", index=False)


def f2():
    dfpath = Path(
        "/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/s103_out/csvs/s103_pseudo_stats.csv"
    )
    df = pd.read_csv(dfpath)
    del df["masked"]
    df.to_csv(dfpath, index=False)


def create_set():
    ##### Gallery requirements
    # Has multiple tracks
    # Priortize: unmask only + low pose + high mag
    #### Probe
    # Mate-searching: the rest of track
    # Nonmate-seaching:
    #   - tracks with 1 image
    #   - tracks not in gallery
    dfpath = Path(
        "/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/s103_out/csvs/s103_pseudo_stats.csv"
    )
    df = pd.read_csv(dfpath)
    trackid_to_userid = {}
    for row in df.itertuples():
        trackid_to_userid[row.track_id] = row.id
    # gudshit = df[(df["is_mask"]==0) & (df["yaw"]<30) & (df["pitch"]<30) & (df["mag"]>25)]
    candidate_trackid_for_enroll = df[
        (df["is_mask"] == 0) & (df["yaw"] < 30) & (df["pitch"] < 30) & (df["mag"] > 25)
    ]["track_id"]

    trackid_for_enroll = set()
    userid_to_trackid = defaultdict(set)

    for track_id, count in candidate_trackid_for_enroll.value_counts().items():
        if count > 3:
            trackid_for_enroll.add(track_id)
            userid_to_trackid[trackid_to_userid[track_id]].add(track_id)
    ids_for_enroll = set(
        [trackid_to_userid[track_id] for track_id in trackid_for_enroll]
    )

    # remove some enroll trackid for mate_searching
    for user_id, track_ids in userid_to_trackid.items():
        n = len(track_ids)
        if n == 1:
            ids_for_enroll.remove(user_id)
        else:
            if n <= 3:
                r = 1
            elif n <= 5:
                r = 2
            else:
                r = n - 3
            tid_to_mates = random.sample(track_ids, r)
            for tid_to_mate in tid_to_mates:
                trackid_for_enroll.remove(tid_to_mate)

    df = df[df["mag"] > 25]
    for row in tqdm(df.itertuples(), total=len(df)):
        i = row.Index
        if row.track_id in trackid_for_enroll:
            df.at[i, "type"] = "enroll"
        else:
            df.at[i, "type"] = "verify"
        if (row.id in ids_for_enroll) and (row.track_id not in trackid_for_enroll):
            df.at[i, "verify_type"] = "mate"
        if row.id not in ids_for_enroll:
            df.at[i, "verify_type"] = "non-mate"
        if df.at[i, "type"] == "enroll":
            df.at[i, "verify_type"] = "N/A"
    # print(candidate_trackid_for_enroll.value_counts())
    print("#user for enroll ", len(set(df[df["type"] == "enroll"]["id"])))
    print("#mate search ", len(set(df[df["verify_type"] == "mate"]["track_id"])))
    print("#nonmate search ", len(set(df[df["verify_type"] == "non-mate"]["track_id"])))
    df.to_csv(
        "/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/s103_out/csvs/s103_idset_v2.csv",
        index=False,
    )
    # print(len(candidate_trackid_for_enroll))


@app.command()
def rename_folders(
    input_path: Path = typer.Argument(..., help="input folder"),
    rename_date: bool = typer.Option(
        True, help="true -> rename date, false -> rename cam"
    ),
):
    date = {
        "20220808070000-20220808095959": "20220808",
        "20220809070000-20220809095959": "20220809",
        "20220810070000-20220810095959": "20220810",
        "20220811070000-20220811095959": "20220811",
        "20220812070000-20220812095959": "20220812",
        "20220814070000-20220814095959": "20220814",
        "20220815070000-20220815095959": "20220815",
        "20220816070000-20220816095959": "20220816",
        "20220817070000-20220817095959": "20220817",
        "20220817080350-20220817145948": "20220817",
    }
    cam = {
        "IP Camera14_BL_S103_BL_S103": "S103",
        "IP Camera15_BL_TNP2_BL_TNP2": "TNP",
        "IP Camera20_172.22.25.26_BL_S102_1": "S102",
        "IP Camera22_172.22.23.34_172.22.23.34": "S107",
        "IP Camera9_172.22.23.28_172.22.23.28": "S106",
        "IP Camera9_BL2_BL2-S101": "S101",
        "IP Camera9_BL_S105_BL_S105": "S105",
    }
    if rename_date:
        folders = os.listdir(input_path)
        for folder in folders:
            if folder in date:
                os.rename((input_path / folder), input_path / date[folder])
    else:
        datefolders = os.listdir(input_path)
        for datefolder in datefolders:
            p = input_path / datefolder
            camfolders = os.listdir(p)
            for camfolder in camfolders:
                newname = camfolder
                for key in cam.keys:
                    if key in camfolder:
                        newname = cam[key]
                finalp = p / newname
                counter = 1
                while finalp.exists():
                    finalp = p / f"{newname}_{counter}"
                    counter += 1
                os.rename((p / camfolder), finalp)


@app.command()
def restructure_cam(
    input_path: Path = typer.Argument(..., help="out"),
    output_path: Path = typer.Argument(..., help="out"),
):
    dates = os.listdir(input_path)
    for date in tqdm(dates):
        datefolder = input_path / date
        cams = os.listdir(datefolder)
        for cam in tqdm(cams):
            realcamname = cam.split("_")[0]
            camoutpath = output_path / realcamname / date
            campath = datefolder / cam / "track_asset"
            trackids = os.listdir(campath)
            for trackid in tqdm(trackids):
                outtrackidpath = camoutpath / trackid
                if outtrackidpath.exists():
                    maxtrackid = max(list(map(int, os.listdir(camoutpath))))
                    outtrackidpath = camoutpath / f"{maxtrackid + 1}"
                shutil.copytree(campath / trackid, outtrackidpath)


@app.command()
def flatten_and_preprocess(input_path: Path = typer.Argument(..., help="inp")):
    dates = os.listdir(input_path)
    for date in dates:
        trackids = os.listdir(input_path / date)
        for trackid in trackids:
            nfiles = len(list(os.listdir(input_path / date / trackid)))
            if nfiles <= 3:
                shutil.rmtree(input_path / date / trackid)
                continue
            new_trackid = f"{date}_{trackid}"
            shutil.move(input_path / date / trackid, input_path / new_trackid)
        datepath = input_path / date
        datepath.rmdir()


@app.command()
def create_cropped_metadata(input_path: Path = typer.Argument(..., help="inp")):
    jpgs = input_path.rglob("*.jpg")
    ids = []
    cropped_img_paths = []
    image_uuids = []
    is_maskeds = []
    trackids = []
    for jpg in jpgs:
        ids.append(jpg.parent.name)
        cropped_img_paths.append(jpg.relative_to(input_path).as_posix())
        image_uuids.append(shortuuid.uuid())
        is_maskeds.append(False)
        trackids.append(jpg.parent.name)
    df = pd.DataFrame(
        {
            "id": ids,
            "track_id": trackids,
            "cropped_img_path": cropped_img_paths,
            "is_masked": is_maskeds,
            "image_uuid": image_uuids,
        }
    )
    df.to_csv((input_path / "cropped_metadata.csv"), index=False)


@app.command()
def aligned_from_csv(
    csv_path: Path = typer.Argument(..., help="csv path"),
    video_dir: Path = typer.Argument(..., help="video path"),
    metadata_path: Path = typer.Argument(..., help="metadata path"),
    output_dir: Path = typer.Argument(..., help="output path"),
):
    if csv_path.is_dir():
        csv_paths = natsorted(list(csv_path.rglob("*.csv")))
    else:
        csv_paths = [csv_path]
    metadata_df = pd.read_csv(metadata_path)

    dfs = []
    for csv_path in tqdm(csv_paths):
        df = pd.read_csv(csv_path, delimiter="\t")
        df = unify_label_for_track_id(df)
        max_frameid = max(df["image_id"].values)
        video_path = video_dir / (csv_path.parent.stem + ".mp4")
        shorten_name = "_".join(video_path.name.split("_")[2:])
        video_date = metadata_df[metadata_df["video_path"] == shorten_name].date.values[
            0
        ]

        assert video_path.exists(), f"{video_path} is not a valid video path"
        cap = cv2.VideoCapture(video_path.as_posix())

        frame_idx = 1
        while cap.isOpened():
            if frame_idx > max_frameid:
                break

            ret, frame = cap.read()
            if frame is None:
                break
            bboxes = df[df["image_id"] == frame_idx]

            if len(bboxes) > 0:
                for _, bbox in bboxes.iterrows():
                    if math.isnan(bbox["face_left"]):
                        continue
                    label = bbox["label"]
                    trackid = bbox["track_id"]
                    conf_score = bbox["score"]
                    lmk = (
                        np.array(bbox["face_lmks"].split(" "))
                        .astype(np.int32)
                        .reshape(-1, 2)
                    )
                    aligned = norm_crop(frame, lmk)
                    id_name = f"{video_date}_{shorten_name.replace('_','').rstrip('.mp4')}_{trackid}_{label}"
                    counter = 0
                    outpath = output_dir / id_name / f"{frame_idx}_{conf_score:.3f}_{counter}.jpg"
                    outpath.parent.mkdir(exist_ok=True, parents=True)
                    while outpath.exists():
                        outpath = output_dir / id_name / f"{frame_idx}_{counter}.jpg"
                        counter += 1
                    cv2.imwrite(outpath.as_posix(), aligned)
                    df.loc[
                        df["annotation_id"] == bbox["annotation_id"], "aligned"
                    ] = outpath.relative_to(output_dir).as_posix()
                    df.loc[
                        df["annotation_id"] == bbox["annotation_id"], "video_path"
                    ] = video_path.relative_to(video_dir).as_posix()

            frame_idx += 1
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(output_dir / "aligned.csv", index=False)


if __name__ == "__main__":
    app()
