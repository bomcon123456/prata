import torchvision
import cv2

video_path = "/vinai/trungdt21/meva-full/datasets/2018-03-07/16/2018-03-07.16-50-00.16-55-00.admin.G329.r13.avi"
reader = torchvision.io.VideoReader(video_path, "video", num_threads=2)
print(reader.get_metadata())
reader.seek(89.3)
n_frames = 2734 - 2679 + 1
out = cv2.VideoWriter(
    "test.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    30,
    (1920, 1072),
)
for i in range(n_frames):
    frame = next(reader)
    # print(frame["data"].per.shape)
    out.write(frame["data"].permute(1, 2, 0).numpy())
out.release()
