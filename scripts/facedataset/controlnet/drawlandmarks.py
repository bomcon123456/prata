import typer
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

app = typer.Typer()


def select_best_landmarks(face_landmarks):
    if face_landmarks.shape[0] == 1:
        return face_landmarks[0]
    else:
        best_idx = 0
        biggest_area = -1
        for i, landmarks in enumerate(face_landmarks):
            # Extract the x and y coordinates of all 68 facial landmarks
            x_coordinates = [landmarks[i][0] for i in range(face_landmarks.shape[1])]
            y_coordinates = [landmarks[i][1] for i in range(face_landmarks.shape[1])]

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


def draw_landmarks(image, landmarks, color="white", radius=2.5):
    draw = ImageDraw.Draw(image)
    for dot in landmarks:
        x, y = dot
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def parse_wh(wh: str):
    splits = wh.split(",")
    if len(splits) == 1:
        w = h = int(splits[0])
    elif len(splits) == 2:
        w, h = list(map(lambda x: int(x.strip()), splits))
    else:
        raise Exception("Invalid format")
    return int(w), int(h)


def f(npy_file, outdir, npy_dir, w, h):
    landmarks = np.array(np.load(npy_file.as_posix()))
    best_landmark = select_best_landmarks(landmarks)
    out_path = outdir / npy_file.relative_to(npy_dir).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con_img = Image.new("RGB", (w, h), color=(0, 0, 0))
    draw_landmarks(con_img, best_landmark)
    con_img.save(out_path.as_posix())


@app.command()
def main(
    npy_dir: Path = typer.Argument(..., exists=True, dir_okay=True),
    outdir: Path = typer.Argument(..., dir_okay=True),
    img_wh: str = typer.Option("512", help="Original image width/height"),
    nprocs: int = typer.Option(16, help="Nprocs"),
):
    w, h = parse_wh(img_wh)
    npy_files = list(npy_dir.rglob("*.npy"))
    with Pool(nprocs) as p:
        list(
            tqdm(
                p.imap(
                    partial(
                        f,
                        outdir=outdir,
                        npy_dir=npy_dir,
                        w=w,
                        h=h,
                    ),
                    npy_files,
                ),
                total=len(npy_files),
            )
        )


if __name__ == "__main__":
    app()
