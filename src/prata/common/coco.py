from collections import defaultdict
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
        obj = coco_obj["annotations"]
    else:
        obj = coco_obj

    pbar = tqdm(obj)
    pbar.set_description("Reading coco annotations")
    for ann in pbar:
        id = ann["id"]
        fname = ann["file_name"]
        res[id] = fname
    return res