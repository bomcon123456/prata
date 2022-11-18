__all__ = ["create_default_coco", "create_image_info", "create_annotation"]
def create_default_coco(labels, ids=None):
    cats = []

    for i, label in enumerate(labels):
        if ids is None:
            id_ = i+1
        else:
            id_ = ids[i]
        cats.append({
            "id": id_,
            "name": label,
            "supercategory": ""
        })

    return {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": "",
        },
        "categories": cats,
        "images": [],
        "annotations": []
    }


def create_image_info(id, width, height, path):
    image_info = {
        "id": int(id),
        "width": width,
        "height": height,
        "file_name": path,
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": 0,
    }
    return image_info


def create_annotation(id, img_id, cat_id, tlwh, attributes):
    ann = {
        "id": int(id),
        "image_id": int(img_id),
        "category_id": int(cat_id),
        "bbox": tlwh,
        "conf": 1,
        "iscrowd": 0,
        "area": float(tlwh[2] * tlwh[3]),
        "attributes": attributes
    }
    return ann
