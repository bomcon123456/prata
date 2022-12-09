from genericpath import exists
from collections import defaultdict
import typer
import pandas as pd
from tqdm.rich import tqdm
from pathlib import Path
import random
import shutil

def f1():
    basedf_path = Path("/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/split_probe_gallery.csv")
    posedf_path = Path("/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/s103_out/csvs/s103_pose.csv")
    magdf_path = Path("/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/s103_out/csvs/s103_mag.csv")
    maskdf_path = Path("/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/s103_out/csvs/s103_mask.csv")

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
        path = row.cropped_img_path.replace("cropped/","")
        pose = posedf[posedf["fname"]==path]
        mag = magdf[magdf["fname"]==path]
        mask = maskdf[maskdf["fname"]==path]
        assert len(pose) == len(mag) == len(mask) == 1
        df.at[i, "is_mask"] = not bool(mask.iloc[0]["not_mask"])
        df.at[i, "pitch"] = float(pose.iloc[0]["pitch"])
        df.at[i, "yaw"] = float(pose.iloc[0]["yaw"])
        df.at[i, "roll"] = float(pose.iloc[0]["roll"])
        df.at[i, "mag"] = float(mag.iloc[0]["mag"])
    df.to_csv("out.csv", index=False)

def f2():
    dfpath = Path("/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/s103_out/csvs/s103_pseudo_stats.csv")
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
    dfpath = Path("/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/s103_out/csvs/s103_pseudo_stats.csv")
    df = pd.read_csv(dfpath)
    trackid_to_userid = {}
    for row in df.itertuples():
        trackid_to_userid[row.track_id] = row.id
    # gudshit = df[(df["is_mask"]==0) & (df["yaw"]<30) & (df["pitch"]<30) & (df["mag"]>25)]
    candidate_trackid_for_enroll = df[(df["is_mask"]==0) & (df["yaw"]<30) & (df["pitch"]<30) & (df["mag"]>25)]["track_id"]

    trackid_for_enroll = set()
    userid_to_trackid = defaultdict(set)

    for track_id, count in candidate_trackid_for_enroll.value_counts().items():
        if count > 3:
            trackid_for_enroll.add(track_id)
            userid_to_trackid[trackid_to_userid[track_id]].add(track_id)
    ids_for_enroll = set([trackid_to_userid[track_id] for track_id in trackid_for_enroll])

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
        if (row.id not in ids_for_enroll):
            df.at[i, "verify_type"] = "non-mate"
        if df.at[i, "type"] == "enroll":
            df.at[i, "verify_type"] = "N/A"
    # print(candidate_trackid_for_enroll.value_counts())
    print("#user for enroll ", len(set(df[df["type"]=="enroll"]["id"])))
    print("#mate search ", len(set(df[df["verify_type"]=="mate"]["track_id"])))
    print("#nonmate search ", len(set(df[df["verify_type"]=="non-mate"]["track_id"])))
    df.to_csv("/home/ubuntu/workspace/trungdt21/matcher/data/cleaning_data/s103_out/csvs/s103_idset_v2.csv", index=False)
    # print(len(candidate_trackid_for_enroll))

create_set()