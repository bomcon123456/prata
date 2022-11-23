from collections import defaultdict
import numpy as np
from tqdm import tqdm


def get_imageid_to_ann(coco_obj):
    res = defaultdict(list)
    if "annotations" in coco_obj:
        obj = coco_obj["annotations"]
    else:
        obj = coco_obj

    pbar = tqdm(obj)
    pbar.set_description("Reading coco annotations")
    for ann in pbar:
        img_id = ann["image_id"]
        res[img_id].append(ann)
    return res

def get_imageid_to_fname(coco_obj):
    res = {}
    if "images" in coco_obj:
        obj = coco_obj["images"]
    else:
        obj = coco_obj

    pbar = tqdm(obj)
    pbar.set_description("Reading coco annotations")
    for ann in pbar:
        id = ann["id"]
        fname = ann["file_name"]
        res[id] = fname
    return res

def get_ltrb_from_ann(anns):
    ltwhs = np.array(list(map(lambda x: x["bbox"], anns)),dtype=np.float32)
    ltbrs = ltwhs.copy()
    ltbrs[:, 2:] += ltbrs[:, 0:2]
    return ltbrs

def get_label_id_from_categories(categories, labels):
    res = []
    d = {x["name"]:x["id"] for x in categories}
    for label in labels:
        assert label in d, f"Invalid label: {label}"
        res.append(d[label])
    return res