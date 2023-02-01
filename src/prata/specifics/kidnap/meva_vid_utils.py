from pathlib import Path

from decord import VideoReader
from decord import cpu
import numpy as np
import cv2

from .meva_file_utils import Activity


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
def crop_from_activity(video_path: Path, activity: Activity, output_path: Path = None):
    assert video_path.exists()
    timespan = activity.timespan
    frames = []
    cap = cv2.VideoCapture(video_path.as_posix())
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
