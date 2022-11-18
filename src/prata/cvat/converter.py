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
        for item in items:
            frame_id = item["attr"]["frame"]
            frame_name = item["image"]["path"]
            hw = item["image"]["size"]
            boxes = []
            lmks = []
            image_info = create_image_info(frame_id, hw[1], hw[0], frame_name)
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
                box_obj = create_annotation(ann_id, frame_id, box["label_id"],  box["bbox"], attrs)
                ann_obj["annotations"].append(box_obj)
                ann_id += 1
        ann_obj["images"] = sorted(ann_obj["images"], key=lambda x: int(x["id"]))
        with open(json_path, "w") as f:
            json.dump(ann_obj, f, indent=2)
        if save_images:
            for img_obj in ann_obj["images"]:
                src = input_imgpath / img_obj["file_name"]
                dst = cur_outpath / img_obj["file_name"]
                shutil.copy2(src, dst)
        
                
                
                    
                
        

    