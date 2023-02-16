from genericpath import exists
import os
import typer
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil
import json

from .data_utils import list_categories
from prata.cvat.coco_utils import create_default_coco

app = typer.Typer()


@app.command()
def filter(
    input_path: Path = typer.Argument(
        ..., help="input path", exists=True, file_okay=True
    ),
    output_path: Path = typer.Argument(..., help="output path"),
    save_annotated_images_only: bool = typer.Option(True, help="Only keep annotated images")
):
    with open(input_path, "r") as f:
        d = json.load(f)
    result = create_default_coco(list_categories, list(range(len(list_categories))))
    old_cat_id_to_new_cat_id = {}
    for category_obj in d["categories"]:
        if category_obj["name"].lower() not in list_categories:
            continue
        old_cat_id_to_new_cat_id[category_obj["id"]] = list_categories.index(category_obj["name"])
    list_old_category_ids = set(old_cat_id_to_new_cat_id.keys())
    anno_counter = 0
    save_img_ids = set()
    for anno_obj in tqdm(d["annotations"]):
        if anno_obj["category_id"] not in list_old_category_ids:
            continue
        obj = anno_obj.copy()
        del obj["segmentation"]
        obj["id"] = anno_counter
        obj["category_id"] = old_cat_id_to_new_cat_id[obj["category_id"]]
        save_img_ids.add(obj["image_id"])
        anno_counter += 1
        result["annotations"].append(obj)

    if save_annotated_images_only:
        image_id_counter = 0
        save_img_ids = sorted(list(save_img_ids))
        old_img_id_to_new_img_id = {}
        for img_obj in tqdm(d["images"]):
            if img_obj["id"] not in save_img_ids:
                continue
            old_img_id_to_new_img_id[img_obj["id"]] = image_id_counter
            new_img_obj = img_obj.copy()
            new_img_obj["id"] = image_id_counter
            result["images"].append(new_img_obj)
            image_id_counter += 1
        
        for anno_obj in tqdm(result["annotations"]):
            old_image_id = anno_obj["image_id"]
            anno_obj["image_id"] = old_img_id_to_new_img_id[old_image_id]
    else:
        result["images"] = d["images"]
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


@app.command()
def prefix_image_path(
    json_path: Path = typer.Argument(
        ..., help="json path", exists=True
    ),
    output_path: Path = typer.Argument(..., help="output path"),
    prefix: str = typer.Argument(..., help="prefix to the current image path"),
):
    if json_path.is_dir():
        json_paths = json_path.glob("*.json")
        if output_path.is_file():
            output_path = output_path.parent
    else:
        json_paths = [json_path]
    if output_path.is_dir():
        output_path.mkdir(exist_ok=True, parents=True)
    for json_path in json_paths:
        with open(json_path, "r") as f:
            d = json.load(f)
        for i in tqdm(range(len(d["images"]))):
            img_obj = d["images"][i]
            name = img_obj["file_name"]
            d["images"][i]["file_name"] = f"{prefix}{name}"
        if output_path.is_dir():
            cur_out = output_path / json_path.name
        else:
            cur_out = output_path
        with open(cur_out, "w") as f:
            json.dump(d, f, indent=2)

if __name__ == "__main__":
    app()
