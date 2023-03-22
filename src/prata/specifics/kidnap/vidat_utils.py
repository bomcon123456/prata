import json
from pathlib import Path
from typing import Optional, List, Any, Union, Dict
from rich import print
import torchvision

import cv2

from prata.specifics.kidnap.vidat_schema import VidatAnnotationFull


def parse_vidat_annotation(annotation_path: Path):
    assert annotation_path.exists(), f"Invalid annotation path {annotation_path}"
    return VidatAnnotationFull.parse_file(annotation_path)


def cut_one_video(video_path: Path, video_ann_path: Path, video_out_path: Path):
    annotations = parse_vidat_annotation(video_ann_path)
    fps = annotations.annotation.video.fps
    for action in annotations.annotation.actionAnnotationList:
        cap = torchvision.io.VideoReader(video_path.as_posix(), "video", num_threads=1)
        start_frame = int(action.start * fps)
        end_frame = int(action.end * fps)
        timespan = end_frame - start_frame
        action_label = annotations.lookup_action_from_id(action.action)
        output_path = video_out_path / action_label / f"{video_path.stem}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        action_instance_ltrbs = annotations.get_ltrb_from_action(
            action.action, start_frame, end_frame
        )
        for action_instance in action_instance_ltrbs:
            counter = 0
            while output_path.exists():
                output_path = (
                    video_out_path / action_label / f"{video_path.stem}_{counter}.mp4"
                )
                counter += 1
            cap.seek(action.start)
            l, t, r, b = action_instance
            t = max(0, t)
            l = max(0, l)
            r = min(1920, r)
            b = min(1080, b)

            out = cv2.VideoWriter(
                output_path.as_posix(),
                cv2.VideoWriter_fourcc(*"XVID"),
                fps,
                (r - l, b - t),
            )
            for _ in range(timespan):
                frame = next(cap)
                f = frame["data"].permute(1, 2, 0).numpy()
                cropped = f[t:b, l:r, :].astype("uint8")
                cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                out.write(cropped)
            out.release()
