from pathlib import Path
import cv2
import numpy as np


def read_widerface(txtfile: Path):
    assert txtfile.exists()
    with open(txtfile, "r") as f:
        lines = f.readlines()

    found = False
    for ext in [".jpg", ".jpeg", ".png"]:
        imgfile = txtfile.parent / (txtfile.stem + ext)
        if imgfile.exists():
            found = True
            break
    assert found, f"{imgfile} not found."
    img = cv2.imread(imgfile.as_posix())
    lines = list(map(lambda x: x.strip().split(), lines))
    lines = np.array(lines, dtype=np.float32)

    for label in lines:
        bbox = label[1:5]
        # print(bbox)
        lmks = label[5:]
        bbox[0::2] *= img.shape[1]
        bbox[1::2] *= img.shape[0]
        bbox[0] -= bbox[2] // 2
        bbox[1] -= bbox[3] // 2
        # print(bbox)
        # exit()
        lmks[0::2] *= img.shape[1]
        lmks[1::2] *= img.shape[0]
        bbox = bbox.astype(np.int32)
        lmks = lmks.astype(np.int32)
        cv2.rectangle(
            img,
            (bbox[0], bbox[1]),
            (bbox[2] + bbox[0], bbox[3] + bbox[1]),
            (0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

        for i in range(5):
            point_x = int(lmks[2 * i])
            point_y = int(lmks[2 * i + 1])
            cv2.circle(img, (point_x, point_y), 2 + 1, clors[i], -1)
    return img
