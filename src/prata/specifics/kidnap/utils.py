from pathlib import Path
import numpy as np
import cv2
import json
from prata.common.coco import get_imageid_to_ann, get_imageid_to_fname
from prata.common.boxes import ious


def crop_for_one_video(input_path: Path, num_negs: int):
    image_basepath = input_path 
    annotation_path = input_path / "annotations/instances_default.json"

    with open(annotation_path, "r") as f:
        obj = json.load(f)
    imageid_to_ann = get_imageid_to_ann(obj)
    imageid_to_fname = get_imageid_to_fname(obj)

    kidnappers, victims, svehicles = get_targets(obj["annotations"])
    target_frames = get_frames_with_targets(kidnappers, victims, svehicles)
    first_interact_frame = get_first_interact_frame(kidnappers, victims)
    print(image_basepath / imageid_to_fname[first_interact_frame])


def get_targets(anns):
    kidnappers = []
    victims = []
    svehicles = []
    for ann in anns:
        if "type" in ann["attributes"]:
            if ann["attributes"]["type"] == "kidnapper":
                kidnappers.append(ann)
            elif ann["attributes"]["type"] == "victim":
                victims.append(ann)
        elif "of_kidnapper" in ann["attributes"]:
            if ann["attributes"]["of_kidnapper"]:
                svehicles.append(ann)
    return kidnappers, victims, svehicles


def get_frames_with_targets(kidnappers, victims, svehicles):
    kidnappers_fids = set(map(lambda x: x["image_id"], kidnappers))
    victim_fids = set(map(lambda x: x["image_id"], victims))
    svehicles_fids = set(map(lambda x: x["image_id"], svehicles))
    min_ = min(*kidnappers_fids, *victim_fids, *svehicles_fids)
    max_ = max(*kidnappers_fids, *victim_fids, *svehicles_fids)

    return list(range(min_, max_ + 1))


def get_first_interact_frame(kidnappers, victims):
    kidnappers_fids = set(map(lambda x: x["image_id"], kidnappers))
    kidnapper_fid2ann = get_imageid_to_ann(kidnappers)
    victim_fids = set(map(lambda x: x["image_id"], victims))
    victim_fid2ann = get_imageid_to_ann(victims)
    init_frame = min(*kidnappers_fids, *victim_fids)
    final_frame = max(*kidnappers_fids, *victim_fids)
    for i in range(init_frame, final_frame + 1):
        if i not in kidnapper_fid2ann or i not in victim_fid2ann:
            continue
        victim = victim_fid2ann[i]
        kidnappers = kidnapper_fid2ann[i]
        ious_ = ious(victim, kidnappers)
        if np.max(ious_) > 0:
            return i
