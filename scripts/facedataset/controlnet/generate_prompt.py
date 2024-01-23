import typer
from typing import Dict, List
import json
from pydantic import BaseModel
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw


class Region(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FaceAttributes(BaseModel):
    age: int
    region: Region
    dominant_gender: str
    race: Dict
    dominant_race: str
    emotion: Dict
    dominant_emotion: str


def select_biggest(attribs: List[FaceAttributes]):
    biggest_area = -1
    biggest_idx = 0
    for i, attrib in enumerate(attribs):
        area = attrib.region.w * attrib.region.h
        if area > biggest_area:
            biggest_area = area
            biggest_idx = i
    return attribs[biggest_idx]


app = typer.Typer()


@app.command()
def main(
    attributes_dir: Path = typer.Argument(..., exists=True, dir_okay=True),
    outpath: Path = typer.Argument(..., file_okay=True),
):
    json_files = list(attributes_dir.rglob("*.json"))
    res = {}
    for json_file in json_files:
        with open(json_file, "r") as f:
            attribs = json.load(f)

        name = json_file.stem
        if "mirror" in name:
            continue
        folder = json_file.parent.name
        splits = name.split("_")
        if folder == "vfhq":
            for i in range(len(splits)):
                if "ypr" in splits[i]:
                    break
            y = int(splits[i].replace("ypr", ""))
            p = int(splits[i + 1])
        elif folder == "celeb":
            for i in range(len(splits)):
                if "yp" in splits[i]:
                    break
            y, p = list(map(int, splits[i + 1 : i + 3]))
        else:
            continue

        attribs = [FaceAttributes.parse_obj(x) for x in attribs]
        if len(attribs) > 1:
            attrib = select_biggest(attribs)
            subject = attrib.dominant_gender.lower()
            emotion = attrib.dominant_emotion.lower() + " "
            race = attrib.dominant_race.lower() + " "
        elif len(attribs) == 0:
            subject = "person"
            emotion = ""
            race = ""
        else:
            attrib = attribs[0]
            subject = attrib.dominant_gender.lower()
            emotion = attrib.dominant_emotion.lower() + " "
            race = attrib.dominant_race.lower() + " "
        # add case norm
        prompt = f"A portrait image of a {emotion}{race}{subject} with head pose: yaw={y}, pitch={p}"
        res[f"{folder}/{name}"] = prompt
        # add case mirror
        name += "_mirror"
        y *= -1
        prompt = f"A portrait image of a {emotion}{race}{subject} with head pose: yaw={y}, pitch={p}"
        res[f"{folder}/{name}"] = prompt

    outpath.parent.mkdir(exist_ok=True, parents=True)
    with open(outpath, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    app()
