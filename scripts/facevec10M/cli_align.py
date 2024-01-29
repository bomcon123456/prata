import random
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import typer
from skimage import transform as trans
from tqdm import tqdm

from prata.utils import get_filelist_and_cache

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

lm478_68_map = {0: 127,
 1: 93,
 2: 132,
 3: 58,
 4: 172,
 5: 150,
 6: 149,
 7: 148,
 8: 152,
 9: 400,
 10: 379,
 11: 364,
 12: 288,
 13: 361,
 14: 366,
 15: 447,
 16: 264,
 17: 156,
 18: 63,
 19: 52,
 20: 65,
 21: 55,
 22: 9,
 23: 336,
 24: 282,
 25: 293,
 26: 383,
 27: 168,
 28: 6,
 29: 195,
 30: 4,
 31: 75,
 32: 99,
 33: 2,
 34: 328,
 35: 460,
 36: 33,
 37: 160,
 38: 158,
 39: 173,
 40: 153,
 41: 144,
 42: 398,
 43: 385,
 44: 387,
 45: 263,
 46: 373,
 47: 380,
 48: 76,
 49: 39,
 50: 37,
 51: 0,
 52: 267,
 53: 269,
 54: 291,
 55: 321,
 56: 314,
 57: 17,
 58: 84,
 59: 91,
 60: 96,
 61: 38,
 62: 13,
 63: 268,
 64: 407,
 65: 317,
 66: 13,
 67: 82
}
lookup_values = np.array(list(lm478_68_map.values()))

def select_best_landmarks(face_landmarks):
    if face_landmarks.shape[0] == 1:
        return face_landmarks[0]
    else:
        best_idx = 0
        biggest_area = -1
        for i, landmarks in enumerate(face_landmarks):
            # Extract the x and y coordinates of all 68 facial landmarks
            x_coordinates = [landmarks[i][0] for i in range(478)]
            y_coordinates = [landmarks[i][1] for i in range(478)]

            # Calculate the minimum and maximum x and y coordinates
            min_x = min(x_coordinates)
            max_x = max(x_coordinates)
            min_y = min(y_coordinates)
            max_y = max(y_coordinates)
            w = max_x - min_x
            h = max_y - min_y
            area = w * h
            if area > biggest_area:
                best_idx = i
                biggest_area = area
        return face_landmarks[best_idx]


def get_lmk(npy_path):
    lm_mediapipe = select_best_landmarks(np.load(npy_path.as_posix()))
    lm = lm_mediapipe[lookup_values]
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack(
        [
            lm[lm_idx[0], :],
            np.mean(lm[lm_idx[[1, 2]], :], 0),
            np.mean(lm[lm_idx[[3, 4]], :], 0),
            lm[lm_idx[5], :],
            lm[lm_idx[6], :],
        ],
        axis=0,
    )
    lm5p = lm5p[[1, 2, 0, 3, 4], :]

    lmk = lm5p.reshape(5, 2)
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


app = typer.Typer(pretty_exceptions_show_locals=False)

def func(image_path, image_dir, lmk_dir, outimagepath, seed, debug_mode=False):
    rela_path = image_path.relative_to(image_dir)
    lmk_path = (lmk_dir / rela_path).with_suffix(".npy")
    assert image_path.exists()
    if not lmk_path.exists():
        return
    out_path = outimagepath / rela_path

    img = cv2.imread(image_path.as_posix())
    lmk = get_lmk(lmk_path)
    if debug_mode:
        out_debug_path = outimagepath.parent / f"debug/{time.time()}.jpg"
        out_debug_path.parent.mkdir(exist_ok=True, parents=True)
        debug_img = img.copy()
        for lmk_ in lmk:
            cv2.circle(debug_img, (int(lmk_[0]), int(lmk_[1])), 1, (0, 0, 255), -1)
        cv2.imwrite(out_debug_path.as_posix(), debug_img)

    aligned = norm_crop(img, lmk)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(out_path.as_posix(), aligned)


@app.command()
def align(
    image_dir: Path = typer.Argument(
        ..., help="Base image_dir", dir_okay=True, exists=True
    ),
    landmark_dir: Path = typer.Argument(
        ..., help="Base lmk_ir", dir_okay=True, exists=True
    ),
    outdir: Path = typer.Argument(..., help="Output dir"),
    nprocs: int = typer.Option(8, help="Num process"),
):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    outimagepath = outdir / "images"
    img_paths = list(get_filelist_and_cache(image_dir, "*.[jp][pn]g"))

    with Pool(nprocs) as p:
        list(
            tqdm(
                p.imap(
                    partial(
                        func,
                        image_dir=image_dir,
                        lmk_dir=landmark_dir,
                        outimagepath=outimagepath,
                        seed=seed,
                    ),
                    img_paths,
                ),
                total=len(img_paths),
                desc="Aligning",
            )
        )

if __name__ == "__main__":
    app()
