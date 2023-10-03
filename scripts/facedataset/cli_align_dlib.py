import glob
from pathlib import Path
import typer
import os
import shutil

import face_alignment
from skimage import io
import cv2
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from tqdm import tqdm
import argparse
import dlib


class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = (
            dlib.get_frontal_face_detector()
        )  # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        if isinstance(image, str):
            img = dlib.load_rgb_image(image)
        else:
            img = image
        dets = self.detector(img, 1)
        for detection in dets:
            face_landmarks = [
                [item.x, item.y]
                for item in self.shape_predictor(img, detection).parts()
            ]
            yield face_landmarks


def get_model(landmarks_model_path):

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False, device="cuda"
    )
    landmarks_detector = LandmarksDetector(landmarks_model_path)

    return fa, landmarks_detector


app = typer.Typer()


@app.command()
def main(
    data_dir: Path = typer.Argument(..., help="data dir", dir_okay=True),
    out_dir: Path = typer.Argument(..., help="out dir", dir_okay=True),
):
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """

    output_dir = out_dir / "lm_facealignment_dlib"
    bad_image_dir = out_dir / "bad"
    output_dir.mkdir(parents=True, exist_ok=True)
    bad_image_dir.mkdir(parents=True, exist_ok=True)
    shape_predictor_path = "./shape_predictor_68_face_landmarks.dat"

    if not os.path.exists(shape_predictor_path):
        raise Exception(
            f"Please download shape predictor ckpt to {shape_predictor_path} according to readme."
        )

    RAW_IMAGES_names = list(data_dir.rglob("*.[jp][pn]g"))
    fa, landmarks_detector = get_model(landmarks_model_path=shape_predictor_path)

    pbar = tqdm(total=len(RAW_IMAGES_names))
    for index, raw_img_path in enumerate(RAW_IMAGES_names):
        pbar.update(1)

        npy_path = output_dir / (raw_img_path.relative_to(data_dir)).with_suffix(".npy")

        if npy_path.exists():
            continue
        npy_path.parent.mkdir(exist_ok=True, parents=True)

        face_landmarks = list(landmarks_detector.get_landmarks(raw_img_path.as_posix()))
        found = False
        if len(face_landmarks) != 0:
            np.save(npy_path.as_posix(), face_landmarks)
            found = True
        else:
            input_img = io.imread(raw_img_path.as_posix())
            if len(input_img.shape) < 3:
                continue
            origin_w, origin_h, _ = input_img.shape
            scale = 1.0
            if max(origin_w, origin_h) > 600:
                scale = 600 / max(origin_w, origin_h)
                input_img = cv2.resize(
                    input_img, (int(origin_h * scale), int(origin_w * scale))
                )
            preds = fa.get_landmarks(input_img)

            if preds is not None:
                found = True
                np.save(npy_path.as_posix(), preds)
        if not found:
            bad_path = bad_image_dir / (raw_img_path.relative_to(data_dir))
            bad_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(raw_img_path, bad_path)


if __name__ == "__main__":
    app()
