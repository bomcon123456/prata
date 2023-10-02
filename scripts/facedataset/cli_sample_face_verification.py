from math import isnan
import pickle
from functools import partial
from multiprocessing import Pool
import typer
from collections import defaultdict
import os
import random
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import shutil
import numpy as np
import json

import cv2
from skimage import transform as trans


arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def get_lmk(lmks5pts, lmks68pts):
    lmk = lmks5pts
    if isinstance(lmk, str):
        lmk = eval(lmk)
    if not isinstance(lmk, list):
        lmk = lmks68pts
        if isinstance(lmk, str):
            lmk = eval(lmk)
        if not isinstance(lmk, list) or len(lmk) != 68 * 3:
            return None
        lmk = np.array(lmk).reshape(3, 68)
        lmk = np.stack(lmk[:2], axis=1).reshape(68, 2).astype(np.int32)
        lefteye = (lmk[36] + lmk[39]) / 2
        righteye = (lmk[42] + lmk[45]) / 2
        nose = lmk[33]
        leftmouth = lmk[48]
        rightmouth = lmk[54]
        lmk = np.array(
            [
                lefteye,
                righteye,
                nose,
                leftmouth,
                rightmouth,
            ],
            dtype=np.int32,
        ).reshape(5, 2)
    return lmk


def estimate_norm(lmk, image_size=112, mode="arcface"):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M


def norm_crop(img, landmark, image_size=112, mode="arcface"):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


app = typer.Typer()


@app.command()
def align_vfhq_full(
    csv_dir: Path = typer.Argument(
        ..., help="Base csv folder", dir_okay=True, exists=True
    ),
    raw_dir: Path = typer.Argument(
        ..., help="Base raw folder", dir_okay=True, exists=True
    ),
    outdir: Path = typer.Argument(..., help="Output dir"),
):
    csv_files = list(csv_dir.rglob("*.csv"))
    outimagepath = outdir / "images"
    fail_images = []

    for csv_file in tqdm(csv_files, desc="Parse CSV"):
        df = pd.read_csv(csv_file)
        id = csv_file.stem
        for row in tqdm(df.itertuples(), total=len(df), desc=f"Parse {csv_file.stem}"):
            frameid = row.frameid
            filename = str(frameid).zfill(8)
            filepath = f"{id}/{filename}.png"
            infilepath = raw_dir / filepath
            out_path = outimagepath / filepath

            if out_path.exists():
                continue

            if not infilepath.exists():
                fail_images.append(infilepath.as_posix())
                continue

            img = cv2.imread(infilepath.as_posix())
            lmk = get_lmk(row.lmks5pts, row.lmks68pts)

            aligned = norm_crop(img, lmk)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(out_path.as_posix(), aligned)
    with open(outdir / "fail_images.txt", "w") as f:
        f.write("\n".join(fail_images))


def func(csv_file, sample_per_csv, raw_dir, outimagepath, seed):
    bin = csv_file.parent.name
    df = pd.read_csv(csv_file)
    real_samples_per_csv = min(len(df), sample_per_csv)
    df = df.sample(n=real_samples_per_csv, random_state=seed)
    id = csv_file.stem
    fail_images = []
    for row in df.itertuples():
        frameid = row.frameid
        filename = str(frameid).zfill(8)
        filepath = f"{id}/{filename}.png"

        infilepath = raw_dir / filepath
        out_path = outimagepath / id / bin / f"{filename}.png"

        if out_path.exists():
            continue

        if not infilepath.exists():
            fail_images.append(infilepath.as_posix())
            continue

        img = cv2.imread(infilepath.as_posix())
        lmk = get_lmk(row.lmks5pts, row.lmks68pts)

        aligned = norm_crop(img, lmk)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(out_path.as_posix(), aligned)
    if len(fail_images) > 0:
        txt_path = outimagepath.parent / "logs" / f"{csv_file.stem}.txt"
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        with open(txt_path, "w") as f:
            f.write("\n".join(fail_images))


@app.command()
def align_vfhq_some(
    csv_dir: Path = typer.Argument(
        ..., help="Base csv folder", dir_okay=True, exists=True
    ),
    raw_dir: Path = typer.Argument(
        ..., help="Base raw folder", dir_okay=True, exists=True
    ),
    outdir: Path = typer.Argument(..., help="Output dir"),
    sample_per_csv: int = typer.Option(50, help="Align how much image per csv"),
    nprocs: int = typer.Option(8, help="Num process"),
):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    CACHE_PATH = Path("~/.cache/trungdt21files/filelist")
    CACHE_PATH.mkdir(exist_ok=True, parents=True)
    filecache = CACHE_PATH / f'{csv_dir.resolve().as_posix().replace("/", "@!@!")}.pkl'
    if filecache.exists():
        with open(filecache, "rb") as f:
            csv_files = pickle.load(f)
    else:
        csv_files = list(csv_dir.rglob("*.csv"))
        with open(filecache, "wb") as f:
            pickle.dump(csv_files, f)
    outimagepath = outdir / "images"

    with Pool(nprocs) as p:
        list(
            tqdm(
                p.imap(
                    partial(
                        func,
                        sample_per_csv=sample_per_csv,
                        raw_dir=raw_dir,
                        outimagepath=outimagepath,
                        seed=seed,
                    ),
                    csv_files,
                ),
                total=len(csv_files),
                desc="Parse CSV",
            )
        )


@app.command()
def main(
    csv_dir: Path = typer.Argument(
        ..., help="Base csv folder", dir_okay=True, exists=True
    ),
    raw_dir: Path = typer.Argument(
        ..., help="Base raw folder", dir_okay=True, exists=True
    ),
    outdir_: Path = typer.Argument(..., help="Output dir"),
    sample_per_bin: int = typer.Option(10, help="How much sample per id_bin"),
):
    # Same ID
    #   Frontal vs Extreme
    #   Frontal vs Frontal
    #   Extreme vs Extreme
    # Different ID
    #   Frontal vs Extreme
    #   Frontal vs Frontal
    #   Extreme vs Extreme
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    frontal_dir = csv_dir / "frontal"
    extreme_dirs = set(os.listdir(csv_dir))
    extreme_dirs.remove("frontal")
    extreme_dirs = list(map(lambda x: csv_dir / x, extreme_dirs))
    frontal_csvs = list(frontal_dir.glob("*.csv"))
    image_counter = 0
    id_to_images = defaultdict(
        lambda: {
            "frontal2frontal": [],
            "frontal2extreme": [],
            "extreme2extreme": [],
            "diff": {},
            "frontals": [],
            "extremes": {},
        }
    )
    final_pairs = []
    outdir = outdir_ / "images"
    outdir.mkdir(exist_ok=True, parents=True)
    for _, csv_path in enumerate(tqdm(frontal_csvs)):
        id = csv_path.stem
        csv_name = csv_path.name
        extreme_csvs = list(
            filter(lambda x: x.exists(), map(lambda x: x / csv_name, extreme_dirs))
        )
        frontal_df = pd.read_csv(csv_path.as_posix())
        extreme_dfs = list(map(lambda x: pd.read_csv(x.as_posix()), extreme_csvs))
        frontal_samples = frontal_df.sample(frac=1, random_state=seed)
        extreme_samples = list(
            map(lambda x: x.sample(frac=1, random_state=seed), extreme_dfs)
        )
        id_image_dir = raw_dir / id
        frontal_names = []

        for frontal_sample in frontal_samples.itertuples():
            if len(frontal_names) > sample_per_bin:
                break
            frameid = frontal_sample.frameid
            filename = str(frameid).zfill(8)
            filepath = id_image_dir / f"{filename}.png"
            frontal_img = cv2.imread(filepath.as_posix())
            lmk = frontal_sample.lmks5pts
            if isinstance(lmk, str):
                lmk = eval(lmk)
            if not isinstance(lmk, list):
                lmk = frontal_sample.lmks68pts
                if isinstance(lmk, str):
                    lmk = eval(lmk)
                if not isinstance(lmk, list) or len(lmk) != 68 * 3:
                    print(f"Skip {frameid} since it doesn't have lmks")
                    continue
                lmk = np.array(lmk).reshape(3, 68)
                lmk = np.stack(lmk[:2], axis=1).reshape(68, 2).astype(np.int32)
                lefteye = (lmk[36] + lmk[39]) / 2
                righteye = (lmk[42] + lmk[45]) / 2
                nose = lmk[33]
                leftmouth = lmk[48]
                rightmouth = lmk[54]
                lmk = np.array(
                    [
                        lefteye,
                        righteye,
                        nose,
                        leftmouth,
                        rightmouth,
                    ],
                    dtype=np.int32,
                ).reshape(5, 2)

            frontal_aligned = norm_crop(frontal_img, lmk)
            frontal_out_path = outdir / f"{str(image_counter).zfill(8)}.png"
            cv2.imwrite(frontal_out_path.as_posix(), frontal_aligned)
            frontal_names.append(frontal_out_path.name)
            image_counter += 1

        extreme_names = defaultdict(list)
        for extreme_path, extreme_samples in zip(extreme_dirs, extreme_samples):
            extreme_bin = extreme_path.name
            tmp_extreme_count = 0
            for extreme_sample in extreme_samples.itertuples():
                if tmp_extreme_count > sample_per_bin:
                    break
                frameid = extreme_sample.frameid
                filename = str(frameid).zfill(8)
                filepath = id_image_dir / f"{filename}.png"
                if not filepath.exists():
                    print(f"{filepath} for extreme doesn't exist.")
                    continue
                extreme_img = cv2.imread(filepath.as_posix())
                lmk = extreme_sample.lmks5pts
                if isinstance(lmk, str):
                    lmk = eval(lmk)
                if not isinstance(lmk, list):
                    lmk = extreme_sample.lmks68pts
                    if isinstance(lmk, str):
                        lmk = eval(lmk)
                    if not isinstance(lmk, list) or len(lmk) != 68 * 3:
                        print(f"Skip {frameid} since it doesn't have lmks")
                        continue
                    lmk = np.array(lmk).reshape(3, 68)
                    lmk = np.stack(lmk[:2], axis=1).reshape(68, 2).astype(np.int32)
                    lefteye = (lmk[36] + lmk[39]) / 2
                    righteye = (lmk[42] + lmk[45]) / 2
                    nose = lmk[33]
                    leftmouth = lmk[48]
                    rightmouth = lmk[54]
                    lmk = np.array(
                        [
                            lefteye,
                            righteye,
                            nose,
                            leftmouth,
                            rightmouth,
                        ],
                        dtype=np.int32,
                    ).reshape(5, 2)

                extreme_aligned = norm_crop(extreme_img, lmk)
                extreme_out_path = outdir / f"{str(image_counter).zfill(8)}.png"
                cv2.imwrite(extreme_out_path.as_posix(), extreme_aligned)
                extreme_names[extreme_bin].append(extreme_out_path.name)
                tmp_extreme_count += 1
                image_counter += 1

        id_to_images[id]["frontals"] = frontal_names
        id_to_images[id]["extremes"] = extreme_names
        # create frontal pair
        indexes = list(range(0, len(frontal_names)))
        np.random.shuffle(indexes)
        frontal_pairs = []
        frontal_extreme_pairs = []
        extreme_extreme_pairs = []
        for i in range(0, len(indexes), 2):
            try:
                frontal_pairs.append((frontal_names[i], frontal_names[i + 1], 1))
            except IndexError:
                break
        # create frontal-extreme pairs
        for extreme_bin, extreme_bin_names in extreme_names.items():
            random.shuffle(frontal_names)
            random.shuffle(extreme_bin_names)
            for frontal_name, extreme_name in zip(frontal_names, extreme_bin_names):
                frontal_extreme_pairs.append((frontal_name, extreme_name, 1))
        # create extreme-extreme pairs
        for extreme_bin in extreme_names.keys():
            for other_extreme_bin in extreme_names.keys():
                if other_extreme_bin == extreme_bin:
                    continue
                extremenames_tmp = extreme_names[extreme_bin]
                random.shuffle(extremenames_tmp)
                otherextremenames_tmp = extreme_names[other_extreme_bin]
                random.shuffle(otherextremenames_tmp)
                for extreme_name, other_extreme_name in zip(
                    extremenames_tmp, otherextremenames_tmp
                ):
                    extreme_extreme_pairs.append((extreme_name, other_extreme_name, 1))
        # add to final
        final_pairs += frontal_pairs + frontal_extreme_pairs + extreme_extreme_pairs
        id_to_images[id]["frontal2frontal"] = frontal_pairs
        id_to_images[id]["frontal2extreme"] = frontal_extreme_pairs
        id_to_images[id]["extreme2extreme"] = extreme_extreme_pairs

    for id1 in id_to_images.keys():
        for id2 in id_to_images.keys():
            if id1 == id2:
                continue

            frontal_pairs = []
            extreme_pairs = []

            # frontal-frontal pair
            frontal_samples_id1 = id_to_images[id1]["frontals"]
            frontal_samples_id2 = id_to_images[id2]["frontals"]
            extreme_samples_id1 = []
            for k, v in id_to_images[id1]["extremes"].items():
                extreme_samples_id1 += v
            extreme_samples_id2 = []
            for k, v in id_to_images[id2]["extremes"].items():
                extreme_samples_id2 += v

            random.shuffle(frontal_samples_id1)
            random.shuffle(frontal_samples_id2)
            # create frontal pair
            for p1, p2 in zip(frontal_samples_id1, frontal_samples_id2):
                frontal_pairs.append((p1, p2, 0))
            # create random pair
            random.shuffle(extreme_samples_id1)
            random.shuffle(extreme_samples_id2)
            for i in range(10):
                extreme_pairs.append(
                    (extreme_samples_id1[i], extreme_samples_id2[i], 0)
                )
                final_pairs += frontal_pairs + extreme_pairs
                id_to_images[id1]["diff"][id2] = dict(
                    frontals=frontal_pairs, extremes=extreme_pairs
                )

    pair_path = outdir_ / "pairs.txt"
    lookup_path = outdir_ / "lookup.json"
    with open(pair_path, "w") as f:
        for pair in final_pairs:
            f.write(f"{pair[0]} {pair[1]} {pair[2]}\n")
    with open(lookup_path, "w") as f:
        json.dump(id_to_images, f, indent=2)


# @app.command()
#     csv_dir: Path = typer.Argument(
# def main(
#         ..., help="Base csv folder", dir_okay=True, exists=True
#     ),
#     raw_dir: Path = typer.Argument(
#         ..., help="Base raw folder", dir_okay=True, exists=True
#     ),
#     outdir: Path = typer.Argument(..., help="Output dir"),
#     sample_per_pair: int = typer.Option(10, help="How much sample per id_bin"),
# ):
#     # Same ID
#     #   Frontal vs Extreme
#     #   Frontal vs Frontal
#     #   Extreme vs Extreme
#     # Different ID
#     #   Frontal vs Extreme
#     #   Frontal vs Frontal
#     #   Extreme vs Extreme
#     seed = 0
#     random.seed(seed)
#     np.random.seed(seed)
#
#     pose_dirs = set(os.listdir(csv_dir))
#     extreme_pose_dirs = pose_dirs.copy()
#     extreme_pose_dirs.remove("frontal")
#     frontal_path = csv_dir / "frontal"
#
#     name_lookup = {}
#     counter = 0
#     pairs = []
#     for id in os.listdir(frontal_path):
#         id = Path(id)
#         csv_path = frontal_path / id
#         frontal_df = pd.read_csv(csv_path)
#         id_img_counter = 0
#         frontal_ids = set()
#         extreme_ids = set()
#         for extreme_pose_dir in extreme_pose_dirs:
#             extreme_csv_path = csv_dir / extreme_pose_dir / id
#             extreme_df = pd.read_csv(extreme_csv_path)
#             samples = extreme_df.sample(n=sample_per_pair, random_state=seed)
#             for extreme_sample in samples.itertuples():
#                 if id_img_counter > len(frontal_df):
#                     idx = id_img_counter % len(frontal_df)
#                 else:
#                     idx = id_img_counter
#                 frontal_sample = frontal_df.loc[idx]
#                 frontal_filename = (
#                     raw_dir / id.stem / f"{str(frontal_sample.frameid).zfill(8)}.png"
#                 )
#                 extreme_filename = (
#                     raw_dir / id.stem / f"{str(extreme_sample.frameid).zfill(8)}.png"
#                 )
#                 frontal_img = cv2.imread(frontal_filename.as_posix())
#                 frontal_lmk = np.array(frontal_sample.lmks5pts).reshape(2, 5)
#                 extreme_img = cv2.imread(extreme_filename.as_posix())
#                 extreme_lmk = np.array(extreme_sample.lmks5pts).reshape(2, 5)
#                 frontal_aligned = norm_crop(frontal_img, frontal_lmk)
#                 extreme_aligned = norm_crop(extreme_img, extreme_lmk)
#                 frontal_out_path = outdir / f"{str(counter).zfill(8)}.png"
#                 extreme_out_path = outdir / f"{str(counter).zfill(8)}.png"
#                 cv2.imwrite(frontal_out_path.as_posix(), frontal_aligned)
#                 cv2.imwrite(extreme_out_path.as_posix(), extreme_aligned)
#                 frontal_ids.add(frontal_out_path.name)
#                 extreme_ids.add(extreme_out_path.name)
#                 pairs.append((frontal_out_path.name, extreme_out_path.name, 1))
#
#                 name_lookup[frontal_out_path.name] = frontal_filename.relative_to(
#                     raw_dir
#                 ).as_posix()
#                 name_lookup[extreme_out_path.name] = extreme_filename.relative_to(
#                     raw_dir
#                 ).as_posix()
#
#                 id_img_counter += 1
#                 id_img_counter += 1


if __name__ == "__main__":
    app()
