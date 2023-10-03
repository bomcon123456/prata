from math import isnan
from functools import lru_cache
import pickle
from functools import partial
from multiprocessing import Pool
from re import sub
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


@lru_cache(2048)
def cached_listdir(d: Path, image_only=False):
    fs = os.listdir(d)
    if image_only:
        fs = [
            file
            for file in fs
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
        ]
    return fs


@lru_cache(2048)
def cached_rglob(d: Path, filter_str=None):
    fs = list(d.rglob("*.[jp][pn]g"))
    if filter_str is not None and isinstance(filter_str, str):
        fs = list(filter(lambda x: filter_str in x.resolve().as_posix(), fs))
    return fs


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


@app.command()
def sample(
    aligned_dir: Path = typer.Argument(..., help="Aligned dirs"),
    outpath: Path = typer.Argument(..., help="output txt file path", file_okay=True),
    n_pairs: int = typer.Option(50000, help="How much pair gonna create"),
    seed: int = typer.Option(0, help="seed"),
):
    random.seed(seed)
    np.random.seed(seed)
    ps = {
        "frontal2frontal": 0.25,
        "frontal2extreme": 0.25,
        "extreme2extreme": 0.5,
    }
    n_samples_per_bin = {bin: int((n_pairs * p) // 2) for bin, p in ps.items()}
    for bin, n_samples in n_samples_per_bin.items():
        print(f"Sample {n_samples} (x2) for bin: {bin}")

    # Get pose info for each ID
    ids = list(cached_listdir(aligned_dir))
    ids_with_frontal_only = []
    ids_with_profile_only = []
    ids_with_frontal_and_profile = []
    n_frontal = 0
    n_extreme = 0
    for id in tqdm(ids, desc="Counting"):
        dir_path = aligned_dir / id
        subdir = list(cached_listdir(dir_path))
        if len(subdir) == 1:
            if subdir[0] == "frontal":
                ids_with_frontal_only.append(id)
                n_frontal += 1
            elif "profile" in subdir[0]:
                ids_with_profile_only.append(id)
                n_extreme += 1
        else:
            if "frontal" in subdir:
                ids_with_frontal_and_profile.append(id)
                n_frontal += 1
            else:
                ids_with_profile_only.append(id)
            n_extreme += 1

    print(f"Number of ID has frontal: {n_frontal}")
    print(f"Number of ID has profile: {n_extreme}")
    print(f"Number of ID has frontal only: {len(ids_with_frontal_only)}")
    print(f"Number of ID has profile only: {len(ids_with_profile_only)}")
    print(f"Number of ID has frontal+profile: {len(ids_with_frontal_and_profile)}")
    frontal_ids = list(set(ids_with_frontal_only + ids_with_frontal_and_profile))
    profile_ids = list(set(ids_with_profile_only + ids_with_frontal_and_profile))
    assert n_frontal == len(frontal_ids)
    assert n_extreme == len(profile_ids)

    # Sample frontal2frontal same ID
    frontal2frontal_sameid_pairs = []
    n_pair_per_id = n_samples_per_bin["frontal2frontal"] // n_frontal
    n_pair_per_id_remain = n_samples_per_bin["frontal2frontal"] % n_frontal
    pbar = tqdm(frontal_ids, desc="Sample frontal2frontal sameID")
    for id in pbar:
        pbar.set_description(
            f"[SameID] frontal2frontal: {len(frontal2frontal_sameid_pairs)}/{n_samples_per_bin['frontal2frontal']}"
        )
        dir_path = aligned_dir / id / "frontal"
        image_files = cached_listdir(dir_path, image_only=True)

        n_samples_for_this_id = n_pair_per_id
        if n_pair_per_id_remain > 0:
            if n_pair_per_id_remain % 2 == 1:
                n_samples_for_this_id += 1
                n_pair_per_id_remain -= 1
            else:
                n_samples_for_this_id += n_pair_per_id_remain // 2
                n_pair_per_id_remain /= 2

        for _ in range(int(n_samples_for_this_id)):
            # Randomly select two images from the folder
            pair = random.sample(image_files, 2)
            # Add the image paths to the list
            pairpath = list(
                map(lambda x: (dir_path / x).relative_to(aligned_dir), pair)
            )
            frontal2frontal_sameid_pairs.append(pairpath)
        if len(frontal2frontal_sameid_pairs) >= n_samples_per_bin["frontal2frontal"]:
            break

    # Sample frontal2extreme same ID
    f2e_sameid_pairs = []
    n_pair_per_id = n_samples_per_bin["frontal2extreme"] // len(
        ids_with_frontal_and_profile
    )
    n_pair_per_id_remain = n_samples_per_bin["frontal2extreme"] % len(
        ids_with_frontal_and_profile
    )
    pbar = tqdm(ids_with_frontal_and_profile, desc="Sample frontal2extreme same ID")
    for id in pbar:
        pbar.set_description(
            f"[SameID] frontal2extreme: {len(f2e_sameid_pairs)}/{n_samples_per_bin['frontal2extreme']}"
        )
        id_path = aligned_dir / id
        image_paths = cached_rglob(id_path)
        frontal_images = list(
            filter(lambda x: "frontal" in x.resolve().as_posix(), image_paths)
        )
        profile_images = list(
            filter(lambda x: "frontal" not in x.resolve().as_posix(), image_paths)
        )
        n_samples_for_this_id = n_pair_per_id
        if n_pair_per_id_remain > 0:
            if n_pair_per_id_remain % 2 == 1:
                n_samples_for_this_id += 1
                n_pair_per_id_remain -= 1
            else:
                n_samples_for_this_id += n_pair_per_id_remain // 2
                n_pair_per_id_remain /= 2

        for _ in range(int(n_samples_for_this_id)):
            image1 = random.choice(frontal_images).relative_to(aligned_dir)
            image2 = random.choice(profile_images).relative_to(aligned_dir)
            f2e_sameid_pairs.append((image1, image2))
        if len(f2e_sameid_pairs) >= n_samples_per_bin["frontal2extreme"]:
            break

    # Sample extreme2extreme same ID
    e2e_sameid_pairs = []
    n_pair_per_id = n_samples_per_bin["extreme2extreme"] // len(profile_ids)
    n_pair_per_id_remain = n_samples_per_bin["extreme2extreme"] % len(profile_ids)
    pbar = tqdm(profile_ids, desc="Sample extreme2extreme same ID")
    for id in pbar:
        pbar.set_description(
            f"[SameID] extreme2extreme: {len(e2e_sameid_pairs)}/{n_samples_per_bin['extreme2extreme']}"
        )
        id_path = aligned_dir / id
        profile_images = cached_rglob(id_path, filter_str="profile")
        if len(profile_images) < 2 or len(profile_images) < n_pair_per_id // 2:
            n_pair_per_id_remain += n_pair_per_id
            continue
        n_samples_for_this_id = n_pair_per_id
        if n_pair_per_id_remain > 0:
            if n_pair_per_id_remain % 2 == 1:
                n_samples_for_this_id += 1
                n_pair_per_id_remain -= 1
            else:
                n_samples_for_this_id += n_pair_per_id_remain // 2
                n_pair_per_id_remain /= 2

        for _ in range(int(n_samples_for_this_id)):
            # Randomly select two images from the folder
            pair = random.sample(profile_images, 2)
            # Add the image paths to the list
            pairpath = list(
                map(lambda x: (dir_path / x).relative_to(aligned_dir), pair)
            )
            e2e_sameid_pairs.append(pairpath)
        if len(e2e_sameid_pairs) >= n_samples_per_bin["extreme2extreme"]:
            break

    # Sample frontal2frontal diff ID
    f2f_diffid_pairs = []
    pbar = tqdm(
        range(n_samples_per_bin["frontal2frontal"]),
        desc="Sample frontal2frontal diffID",
    )
    for _ in pbar:
        pbar.set_description(
            f"[DiffID] frontal2frontal: {len(f2f_diffid_pairs)}/{n_samples_per_bin['frontal2frontal']}"
        )
        id1, id2 = random.sample(frontal_ids, 2)
        id1_path = aligned_dir / id1 / "frontal"
        id2_path = aligned_dir / id2 / "frontal"
        id1_files = cached_listdir(id1_path, image_only=True)
        id2_files = cached_listdir(id2_path, image_only=True)
        pair = random.sample(id1_files, 1) + random.sample(id2_files, 1)
        pairpath = [
            (id1_path / pair[0]).relative_to(aligned_dir),
            (id2_path / pair[1]).relative_to(aligned_dir),
        ]
        f2f_diffid_pairs.append(pairpath)

    # Sample frontal2extreme diff ID
    f2e_diffid_pairs = []
    pbar = tqdm(
        range(n_samples_per_bin["frontal2extreme"]),
        desc="Sample frontal2frontal diffID",
    )
    for _ in pbar:
        pbar.set_description(
            f"[DiffID] frontal2extreme: {len(f2e_diffid_pairs)}/{n_samples_per_bin['frontal2extreme']}"
        )
        id1, id2 = random.sample(frontal_ids, 1) + random.sample(profile_ids, 1)
        while id1 == id2:
            id1, id2 = random.sample(frontal_ids, 1) + random.sample(profile_ids, 1)
        id1_path = aligned_dir / id1 / "frontal"
        id2_path = aligned_dir / id2
        id1_files = cached_listdir(id1_path, image_only=True)
        id2_files = cached_rglob(id2_path, filter_str="profile")
        pair = random.sample(id1_files, 1) + random.sample(id2_files, 1)
        pairpath = list(map(lambda x: (id1_path / x).relative_to(aligned_dir), pair))
        f2e_diffid_pairs.append(pairpath)

    # Sample extreme2extreme diff ID
    e2e_diffid_pairs = []
    pbar = tqdm(
        range(n_samples_per_bin["extreme2extreme"]),
        desc="Sample frontal2frontal diffID",
    )
    for _ in pbar:
        pbar.set_description(
            f"[DiffID] extreme2extreme: {len(e2e_diffid_pairs)}/{n_samples_per_bin['extreme2extreme']}"
        )
        id1, id2 = random.sample(profile_ids, 2)
        id1_path = aligned_dir / id1
        id2_path = aligned_dir / id2
        id1_files = cached_rglob(id1_path, filter_str="profile")
        id2_files = cached_rglob(id2_path, filter_str="profile")
        pair = random.sample(id1_files, 1) + random.sample(id2_files, 1)
        pairpath = [
            (id1_path / pair[0]).relative_to(aligned_dir),
            (id2_path / pair[1]).relative_to(aligned_dir),
        ]
        e2e_diffid_pairs.append(pairpath)

    final_pairs = []
    sets = set()
    for pair in frontal2frontal_sameid_pairs + f2e_sameid_pairs + e2e_sameid_pairs:
        pair = tuple(map(str, pair))
        if pair in sets:
            continue
        sets.add(pair)
        final_pairs.append(f"{pair[0]}\t{pair[1]}\t1")
    for pair in f2f_diffid_pairs + f2e_diffid_pairs + e2e_diffid_pairs:
        pair = tuple(map(str, pair))
        if pair in sets:
            continue
        sets.add(pair)
        final_pairs.append(f"{pair[0]}\t{pair[1]}\t0")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        f.write("\n".join(final_pairs))


if __name__ == "__main__":
    app()
