from pathlib import Path
from typing import List
import json

from decord import VideoReader
from decord import cpu
import numpy as np
import cv2
import torchvision

from .meva_file_utils import Activity, NumpyEncoder


def crop_from_activities(
    video_path: Path,
    activities: List[Activity],
    engine: str,
    cached: bool,
    output_path: Path = None,
    filter_set=None,
):
    frames = None
    fps = None
    if engine == "opencv":
        cap = cv2.VideoCapture(video_path.as_posix())
        fnc = crop_from_activity_opencv
        fps = cap.get(cv2.CAP_PROP_FPS)
        if len(activities) > 1 and cached:
            pass
    elif engine == "torch":
        cap = torchvision.io.VideoReader(video_path, "video", num_threads=2)
        fps = cap.get_metadata()["fps"]
        fnc = crop_from_activity_torch
        if len(activities) > 1 and cached:
            pass
    use_cached = len(activities) > 1 and cached

    for activity in activities:
        activity: Activity
        activity_type = activity.class_name
        if filter_set is not None and activity_type.lower() not in filter_set:
            continue
        activity_basepath = output_path / activity_type
        num_vids = len(list(activity_basepath.glob(f"{video_path.stem}*.avi")))
        outvideo_path = activity_basepath / f"{video_path.stem}_{num_vids+1}.avi"
        outjson_path = activity_basepath / f"{video_path.stem}_{num_vids+1}.json"
        outvideo_path.parent.mkdir(parents=True, exist_ok=True)
        if use_cached:
            assert frames is not None
            crop_from_activity_cached(frames, fps, activity, output_path)
        else:
            fnc(cap, activity, output_path)
        activity_dict = activity.dict()
        activity_dict["bounded_tlbr"] = activity.get_bounded_tlbr().tolist()
        with open(outjson_path, "w") as f:
            json.dump(activity_dict, f, indent=2, cls=NumpyEncoder)


def crop_from_activity_torch(cap, activity: Activity, output_path: Path = None):
    cap: torchvision.io.VideoReader


def crop_from_activity_opencv(cap, activity: Activity, output_path: Path = None):
    timespan = activity.timespan
    frames = []
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # check for valid frame number
    if timespan[0] >= 0 & timespan[0] <= totalFrames:
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, timespan[0])
    for i in range(timespan[0], timespan[1] + 1):
        ret, frame = cap.read()
        frames.append(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = np.array(frames)
    t, l, b, r = activity.get_bounded_tlbr().astype(np.int32)
    cropped = frames[:, t:b, l:r].astype("uint8")

    if output_path:
        out = cv2.VideoWriter(
            output_path.as_posix(),
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (cropped.shape[2], cropped.shape[1]),
        )
        for img in cropped:
            out.write(img)
        out.release()
    # cv2.destroyAllWindows()

    return cropped


def crop_from_activity_cached(
    all_frames: np.ndarray, fps: int, activity: Activity, output_path: Path = None
):
    timespan = activity.timespan
    frames = all_frames[timespan[0] : timespan[1] + 1]
    t, l, b, r = activity.get_bounded_tlbr().astype(np.int32)
    cropped = frames[:, t:b, l:r].astype("uint8")

    if output_path:
        out = cv2.VideoWriter(
            output_path.as_posix(),
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (cropped.shape[2], cropped.shape[1]),
        )
        for img in cropped:
            out.write(img)
        out.release()
    # cv2.destroyAllWindows()

    return cropped


# def crop_from_activity(video_path: Path, activity: Activity, output_path: Path = None):
#     assert video_path.exists()
#     vr = VideoReader(video_path.as_posix(), ctx=cpu(0))
#     timespan = activity.timespan
#     # timestamps = list(range(timespan[0], timespan[1]))
#     # 1. the simplest way is to directly access frames
#     # tout = cv2.VideoWriter(
#     #     "frame.avi",
#     #     cv2.VideoWriter_fourcc(*"XVID"),
#     #     30,
#     #     (1920, 1072),
#     # )
#     frames = []
#     for i in range(len(vr)):
#         frame = vr[i].asnumpy()

#         if i >= timespan[0] and i <= timespan[1]:
#             frames.append(frame)
#             # tout.write(frame)
#         elif i > timespan[1]:
#         #     tout.release()
#             break
#     frames = np.array(frames)
#     t, l, b, r = activity.get_bounded_tlbr().astype(np.int32)
#     cropped = frames[:, t:b, l:r].astype("uint8")

#     if output_path:
#         out = cv2.VideoWriter(
#             output_path.as_posix(),
#             cv2.VideoWriter_fourcc(*"XVID"),
#             vr.get_avg_fps(),
#             (cropped.shape[2], cropped.shape[1]),
#         )
#         for img in cropped:
#             out.write(img)
#         out.release()

#     return cropped
