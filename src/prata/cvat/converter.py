import json
import numpy as np
from tqdm.rich import tqdm
from pathlib import Path
import shutil

from .coco_utils import *
from .utils import max_ioa

def datumaro_to_coco(
    input_path: Path,
    output_path: Path,
    save_images: bool,
    filter_negative: bool=False,
    debug: bool=False
):
    output_path.mkdir(exist_ok=True, parents=True)
    input_paths = []
    if input_path.is_file():
        input_paths.append(input_path)
    elif input_path.is_dir():
        input_paths = list(input_path.rglob("*/annotations/default.json"))
    assert len(input_paths) != 0, "Cant find any json!"
    pbar = tqdm(input_paths)
    for input_path in pbar:
        dataset_name = input_path.parent.parent.parent.name
        pbar.set_description(dataset_name)
        input_imgpath = input_path.parent.parent / "images"
        cur_outpath = output_path / dataset_name
        annotations_path = cur_outpath / "annotations"
        images_path = cur_outpath / "images"
        annotations_path.mkdir(parents=True, exist_ok=True)
        images_path.mkdir(parents=True, exist_ok=True)
        json_path = annotations_path/"instances_default.json"
        
        with open(input_path, "r") as f:
            obj = json.load(f)
        items = obj["items"]
        labels = list(map(lambda x: x["name"], obj["categories"]["label"]["labels"]))
        ann_obj = create_default_coco(labels, list(range(len(labels))))        
        ann_id = 1
        count_frame_id = 1
        original_fid_2_final_fid = {}
        for item in items:
            frame_id = item["attr"]["frame"]
            n_annotations = len(item["annotations"])
            if filter_negative and n_annotations == 0:
                continue
            original_fid_2_final_fid[frame_id] = count_frame_id
            frame_name = item["image"]["path"]
            hw = item["image"]["size"]
            boxes = []
            lmks = []
            image_info = create_image_info(frame_id, hw[1], hw[0], frame_name)
            if filter_negative:
                image_info["id"] = original_fid_2_final_fid[int(image_info["id"])]
            ann_obj["images"].append(image_info)

            for ann in item["annotations"]:
                if ann["type"] == "bbox":
                    boxes.append(ann)
                elif ann["type"] == "points":
                    lmks.append(ann)
            for lmk in lmks:
                if len(lmk["points"]) != 10:
                    continue
                pts = np.array(lmk["points"]).reshape(-1,2)
                min_xy = min(pts[:,0]), min(pts[:,1])
                max_xy = min(pts[:,0]), min(pts[:,1])
                bound_ltrb = [*min_xy, *max_xy]
                founded_box = None
                max_ioa_ = 0
                for b in boxes:
                    ltwh = np.array(b["bbox"])
                    ltrb = ltwh
                    ltrb[2:] += ltwh[:2]
                    ioa = max_ioa(bound_ltrb, ltrb)
                    if debug:
                        print("ioa=", ioa)
                    if ioa > max_ioa_:
                        founded_box = b
                        max_ioa_ = ioa
                assert founded_box is not None, f"Cant find box for landmark! {dataset_name}, frameid={frame_id}"
                founded_box["landmarks"] = lmk["points"]
            for box in boxes:
                attrs = {
                    **box["attributes"]
                }
                if "landmarks" in box:
                    attrs["landmarks"] = box["landmarks"]
                final_frame_id = frame_id if not filter_negative else original_fid_2_final_fid[frame_id]
                box_obj = create_annotation(ann_id, final_frame_id, box["label_id"],  box["bbox"], attrs)
                ann_obj["annotations"].append(box_obj)
                ann_id += 1
            count_frame_id += 1
        ann_obj["images"] = sorted(ann_obj["images"], key=lambda x: int(x["id"]))
        with open(json_path, "w") as f:
            json.dump(ann_obj, f, indent=2)
        if save_images:
            for img_obj in ann_obj["images"]:
                src = input_imgpath / img_obj["file_name"]
                dst = cur_outpath / img_obj["file_name"]
                shutil.copy2(src, dst)
        
def datumaro_to_widerface(
    input_path: Path,
    output_path: Path,
    filter_negative: bool=False,
    debug: bool=False
):
    output_path.mkdir(exist_ok=True, parents=True)
    input_paths = []
    if input_path.is_file():
        input_paths.append(input_path)
    elif input_path.is_dir():
        input_paths = list(input_path.rglob("*/annotations/default.json"))
    assert len(input_paths) != 0, "Cant find any json!"
    pbar = tqdm(input_paths)
    for input_path in pbar:
        dataset_name = input_path.parent.parent.parent.name
        pbar.set_description(dataset_name)
        input_imgpath = input_path.parent.parent / "images"
        cur_outpath = output_path
        
        with open(input_path, "r") as f:
            obj = json.load(f)
        items = obj["items"]
        labels = list(map(lambda x: x["name"], obj["categories"]["label"]["labels"]))
        ann_obj = create_default_coco(labels, list(range(len(labels))))        
        for item in items:
            frame_id = item["attr"]["frame"]
            n_annotations = len(item["annotations"])
            if filter_negative and n_annotations == 0:
                continue
            frame_name = item["image"]["path"]
            hw = item["image"]["size"]
            boxes = []
            lmks = []
            image_info = create_image_info(frame_id, hw[1], hw[0], frame_name)
            if filter_negative:
                # TODO
                pass
            ann_obj["images"].append(image_info)

            for ann in item["annotations"]:
                if ann["type"] == "bbox":
                    boxes.append(ann)
                elif ann["type"] == "points":
                    lmks.append(ann)
            for lmk in lmks:
                if len(lmk["points"]) != 10:
                    continue
                pts = np.array(lmk["points"]).reshape(-1,2)
                min_xy = min(pts[:,0]), min(pts[:,1])
                max_xy = min(pts[:,0]), min(pts[:,1])
                bound_ltrb = [*min_xy, *max_xy]
                founded_box = None
                max_ioa_ = 0
                for b in boxes:
                    box_ltwh = np.array(b["bbox"])
                    box_ltrb = box_ltwh
                    box_ltrb[2:] += box_ltwh[:2]
                    ioa = max_ioa(bound_ltrb, box_ltrb)
                    if debug:
                        print("ioa=", ioa)
                    if ioa > max_ioa_:
                        founded_box = b
                        max_ioa_ = ioa
                assert founded_box is not None, f"Cant find box for landmark! {dataset_name}, frameid={frame_id}"
                founded_box["landmarks"] = lmk["points"]
            img_path = input_imgpath / frame_name
            frame_name = dataset_name + "_" + Path(frame_name).name
            
            out_img_path = cur_outpath / Path(frame_name).name
            out_txt_path = cur_outpath / (Path(frame_name).stem + ".txt")
            if len(boxes) == 0 and filter_negative:
                continue
            shutil.copy2(img_path, out_img_path)
            f = open(out_txt_path, "w")
            for box in boxes:
                box_ltwh = np.array(box["bbox"])
                box_ltrb = box_ltwh
                box_ltrb[2:] += box_ltwh[:2]
                normed_ltrb = box_ltrb.copy()
                normed_ltrb[0::2] /= hw[1]
                normed_ltrb[1::2] /= hw[0]
                label = 0
                if "has_mask" in box["attributes"]:
                    label = 1 if box["attributes"]["has_mask"] else 0

                if "landmarks" in box:
                    lmks = np.array(box["landmarks"], dtype=np.float32)
                    lmks[0::2] /= hw[1]
                    lmks[1::2] /= hw[0]
                    lmks = lmks.tolist()
                else:
                    lmks = [-1] * 10

                bbox = normed_ltrb.tolist()
                line = [label, *bbox, *lmks]
                line = list(map(str, line))
                line = " ".join(line)
                f.write(f"{line}\n")
   