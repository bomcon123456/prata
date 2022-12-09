from pathlib import Path
import random
import numpy as np
import shutil
from tqdm import tqdm
import cv2


def oneclassify_onefolder(input_path: Path, output_path: Path):
    files = list(input_path.rglob("*"))
    for file in tqdm(files):
        out_path = output_path / (file.relative_to(input_path))
        out_path.parent.mkdir(exist_ok=True, parents=True)
        if file.suffix == ".txt":
            fout = open(out_path, "w")
            fin = open(file, "r")
            lines = fin.readlines()
            lines = list(map(lambda x: x.strip().split(), lines))
            # collapse label_id to 0 (one class)
            for line in lines:
                line[0] = "0"
                for number in line[1:]:
                    if float(number) > 1.0:
                        number = "1."
            lines = list(map(" ".join, lines))
            lines = list(map(lambda x: x + "\n", lines))
            fout.writelines(lines)
        else:
            shutil.copy2(file, out_path)


def parse_yolo(txtpath: Path):
    """
        Return [[label, normed_ltbr, normed_lmk]]
    """
    if txtpath.suffix != ".txt":
        raise Exception(f"Invalid txtpath: {txtpath}")
    with open(txtpath, "r") as f:
        lines = f.readlines()

    lines = list(map(lambda x: x.strip().split(), lines))
    for i in range(len(lines)):
        lines[i] = list(map(float, lines[i]))
    annotations = np.array(lines, dtype=np.float).reshape(-1, 15)
    labels = annotations[:, 0]
    normed_ltwh = annotations[:, 1:5]
    normed_ltwh[:, 0] -= normed_ltwh[:, 2] / 2
    normed_ltwh[:, 1] -= normed_ltwh[:, 3] / 2
    normed_ltbr = normed_ltwh
    normed_ltbr[:, 2:] += normed_ltbr[:, :2]
    normed_lmk = annotations[:, 5:]
    d = {
        "annotations": annotations,
        "labels": labels,
        "normed_ltbr": normed_ltbr,
        "normed_lmks": normed_lmk,
    }
    return d


def rescale_normed(input, hw):
    prev_shape = input.shape
    input = input.reshape(-1)
    input[0::2] *= hw[1]
    input[1::2] *= hw[0]
    input = input.reshape(prev_shape)
    return input


def get_crop(
    img,
    ltrb,
    annotations,
    img_height,
    img_width,
    min_size_shift=-0.1,
    max_size_shift=0.1,
    min_pos_shift=-0.1,
    max_pos_shift=0.1,
    min_crop_height=100,
    min_crop_width=100,
    ioa_threshold=0.7,
    ratio_threshold=0.1,
):
    # print(annotations)
    x_min, y_min, x_max, y_max = ltrb
    width, height = x_max - x_min, y_max - y_min

    scale = max(height, width) * (
        1 + (max_size_shift - min_size_shift) * random.random() + min_size_shift
    )

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    x_min = x_center - scale / 2
    y_min = y_center - scale / 2
    x_max = x_center + scale / 2
    y_max = y_center + scale / 2

    x_min -= scale * 1.5
    x_max += scale * 1.5
    y_min -= scale * 0.5
    y_max += scale * 2.5

    shift = ((max_pos_shift - min_pos_shift) * random.random() + min_pos_shift) * scale
    x_min += shift
    x_max += shift
    y_min += shift
    y_max += shift

    if x_min < 0:
        x_max -= x_min
        x_min = 0

    if y_min < 0:
        y_max -= y_min
        y_min = 0

    x_min = int(max(min(x_min, img_width), 0))
    x_max = int(max(min(x_max, img_width), 0))
    y_min = int(max(min(y_min, img_height), 0))
    y_max = int(max(min(y_max, img_height), 0))

    if y_max - y_min != x_max - x_min:
        return None, None

    crop = img[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None, None

    crop_height, crop_width, _ = crop.shape

    if crop_height != crop_width:
        extended_size = min(crop_width, crop_height)
        crop = cv2.copyMakeBorder(
            crop,
            0,
            extended_size - crop_height,
            0,
            extended_size - crop_width,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    crop_height, crop_width, _ = crop.shape
    if crop_height < min_crop_height or crop_width < min_crop_width:
        return None, None

    new_ann = get_annotations_in_crop(
        x_min,
        y_min,
        x_max,
        y_max,
        annotations,
        crop_width,
        crop_height,
        ioa_threshold,
        ratio_threshold,
    )
    new_ann = np.array(new_ann).reshape(-1, 15)
    return (crop, new_ann)

def plot_face(img, ltrb, lmks, inplace=False):
    if inplace:
        viz = img
    else:
        viz = img.copy()
    ltrb = np.array(ltrb).reshape(4)
    lmks = np.array(lmks).reshape(-1, 2)
    cv2.rectangle(viz, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
    for landmark in lmks:
        cv2.circle(viz, (int(landmark[0]), int(landmark[1])), 3 ,(0,0,255), -1)
    return viz

def get_annotations_in_crop(
    x_min,
    y_min,
    x_max,
    y_max,
    annotations,
    crop_width,
    crop_height,
    ioa_threshold,
    ratio_threshold,
):
    res = []
    for candiate_annotation in annotations:
        (
            candiate_x_min,
            candiate_y_min,
            candiate_x_max,
            candiate_y_max,
        ) = candiate_annotation[1:5]
        candidate_width = candiate_x_max - candiate_x_min
        candidate_height = candiate_y_max - candiate_y_min
        annotation_area = candidate_width * candidate_height
        if annotation_area < 1e-5:
            continue

        if (
            candiate_x_min >= x_max
            or candiate_y_min >= y_max
            or x_min >= candiate_x_max
            or y_min >= candiate_y_max
        ):
            continue

        candiate_x_max = candiate_x_min + candidate_width
        candiate_y_max = candiate_y_min + candidate_height

        intersect_xmin = max(candiate_x_min, x_min)
        intersect_xmax = min(candiate_x_max, x_max)
        intersect_ymin = max(candiate_y_min, y_min)
        intersect_ymax = min(candiate_y_max, y_max)

        intersect_area = (intersect_xmax - intersect_xmin) * (
            intersect_ymax - intersect_ymin
        )

        if intersect_area / annotation_area < ioa_threshold:
            continue

        intersect_width = intersect_xmax - intersect_xmin
        intersect_height = intersect_ymax - intersect_ymin
        if (
            min(intersect_width, intersect_height) / min(crop_width, crop_height)
            < ratio_threshold
        ):
            continue

        output_annotation = candiate_annotation.copy()
        output_annotation[1:5] = (
            intersect_xmin - x_min,
            intersect_ymin - y_min,
            intersect_xmin - x_min + intersect_width,
            intersect_ymin - y_min + intersect_height,
        )

        output_annotation[5::2] -= x_min
        output_annotation[6::2] -= y_min
        res.append(output_annotation)
    return res


def cropify_onefolder(input_path: Path, output_path: Path, debug: bool):
    files = list(input_path.rglob("*.txt"))
    output_path.mkdir(exist_ok=True, parents=True)
    for file in tqdm(files):
        ann_obj = parse_yolo(file)
        exts = [".jpg", ".jpeg", ".png"]
        found = False
        for ext in exts:
            img_path = file.parent / f"{file.stem}{ext}"
            if img_path.exists():
                found = True
                break
        assert found, f"{img_path} not exists"

        img = cv2.imread(img_path.as_posix())
        hw = img.shape[:2]
        annotations = ann_obj["annotations"]
        rescaled_annotations = annotations.copy()
        for i, ann in enumerate(rescaled_annotations):
            rescaled_annotations[i, 1:] = rescale_normed(
                rescaled_annotations[i, 1:], hw
            )

        if debug:
            viz = img.copy()
            # print(rescaled_annotations)
            # print("---------------")
            for i, ann in enumerate(rescaled_annotations):
                ltrb = ann[1:5]
                lmks = ann[5:].reshape(-1, 2)
                plot_face(viz, ltrb, lmks, inplace=True)
            p_ = output_path.parent / f"{output_path.stem}_fullframe" / f"{file.stem}.png"
            p_.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(p_.as_posix(), viz)

        for i, ann in enumerate(rescaled_annotations):
            ltrb = ann[1:5]
            crop, crop_ann = get_crop(
                img,
                ltrb,
                rescaled_annotations,
                hw[0],
                hw[1],
                # min_pos_shift=0,
                # max_pos_shift=0,
                # min_size_shift=0,
                # max_size_shift=0,
            )
            if crop is None:
                continue
            p_img = output_path / f"{file.stem}_{i}_n_{len(crop_ann)}.png"
            p_txt = output_path / f"{file.stem}_{i}_n_{len(crop_ann)}.txt"
            if debug:
                for an in crop_ann:
                    plot_face(crop, an[1:5], an[5:], inplace=True)
            fout = open(p_txt, "w") 
            for an in crop_ann:
                tmp = an[1:5].copy()
                wh = tmp[2:] - tmp[:2]
                tmp[0] += wh[0] / 2
                tmp[1] += wh[1] / 2
                tmp[2] = wh[0]
                tmp[3] = wh[1]
                an[1:5] = tmp
                an[1::2] /= crop.shape[1]
                an[2::2] /= crop.shape[0]
                s = [int(an[0]), *an[1:]]
                s = list(map(str,s))
                s = " ".join(s) + "\n"
                fout.write(s)
            fout.close()
            cv2.imwrite(p_img.as_posix(), crop)
            # exit()
            #     p.mkdir(exist_ok=True, parents=True)

            # TODO:
