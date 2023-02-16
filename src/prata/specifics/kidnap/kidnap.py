import typer
from tqdm import tqdm
from natsort import natsorted
import os
import shutil
from pathlib import Path

from prata.specifics.kidnap.utils import crop_for_one_video, create_rider_from_ann
from prata.specifics.kidnap.idd import cut_rider_from_idd_xml

app = typer.Typer()


@app.command()
def split_train_val(
    input_path: Path = typer.Argument(..., help="Path to input"),
    output_path: Path = typer.Option("./out", help="output path"),
):
    val_folders = [
        "kidnap_20220816103233-20220816113000_2",
        "kidnap_20220816104513-20220816114500_0",
        "kidnap_20220816103053-20220816113000_1_kidnap",
        "kidnap_20220816103249-20220816113000_1_kidnap",
        "kidnap_20220816103233-20220816113000_0",
        "kidnap_20220816104513-20220816114500_1_kidnap",
        "kidnap_20220816103249-20220816113000_4",
        "22_20220816105714_20220816113000_8568764",
        "kidnap_20220816104513-20220816114500_2",
        "kidnap_20220816103249-20220816113000_0",
        "kidnap_20220816103249-20220816113000_2",
        "kidnap_20220816103053-20220816113000_2",
        "kidnap_20220816103233-20220816113000_4",
        "kidnap_20220816103233-20220816113000_1_kidnap",
        "kidnap_20220816103233-20220816113000_3_kidnap",
        "kidnap_20220816103249-20220816113000_3_kidnap",
    ]
    val_path = output_path / "val"
    train_path = output_path / "train"
    (output_path / "train").mkdir(exist_ok=True, parents=True)
    (output_path / "val").mkdir(exist_ok=True, parents=True)
    scenes = set(os.listdir(input_path))
    for vf in val_folders:
        p = input_path / vf
        assert p.exists()
        shutil.copytree(p, val_path / vf)
        scenes.remove(vf)
    for tf in scenes:
        p = input_path / tf
        assert p.exists()
        shutil.copytree(p, train_path / tf)


@app.command()
def crop_static_videos(
    video_basepath: Path = typer.Argument(..., help="Base video_path"),
    annotation_basepath: Path = typer.Argument(..., help="Base annotation_path"),
    output_path: Path = typer.Option("./out", help="Output path"),
    num_negs: int = typer.Option(25, help="Number of negative videos generating."),
):
    annotation_paths = list(annotation_basepath.rglob("instances_default.json"))
    output_path.mkdir(exist_ok=True, parents=True)
    pbar = tqdm(natsorted(annotation_paths))
    for annotation_path in pbar:
        video_name = annotation_path.parent.parent.name
        pbar.set_description(f"{video_name}")

        video_path = video_basepath / f"{video_name}.mp4"
        assert video_path.exists(), f"{video_path} not exists!"
        out = output_path / f"{video_name}"
        if out.exists():
            continue
        (out / "pos").mkdir(exist_ok=True, parents=True)
        (out / "neg").mkdir(exist_ok=True, parents=True)
        crop_for_one_video(video_path, annotation_path, out, num_negs)


@app.command()
def create_rider_dataset(
    input_path: Path = typer.Argument(..., help="Base input path"),
    output_path: Path = typer.Option("./rider_dataset", help="output path"),
    get_bikes_only: bool = typer.Option(True, help="get bikes no rider"),
):

    annotation_paths = list(input_path.rglob("instances_default.json"))
    output_path.mkdir(exist_ok=True, parents=True)
    pbar = tqdm(natsorted(annotation_paths))
    for annotation_path in pbar:
        dataset_name = annotation_path.parent.parent.name
        pbar.set_description(dataset_name)
        p = output_path / dataset_name
        p.mkdir(exist_ok=True, parents=True)
        create_rider_from_ann(annotation_path, p, get_bikes_only)


@app.command()
def create_rider_dataset_from_idd(
    annotation_path: Path = typer.Argument(..., help="Base input path"),
    img_path: Path = typer.Argument(..., help="Base img path"),
    output_path: Path = typer.Option("./rider_dataset", help="output path"),
    pad_size: int = typer.Option(15, help="padding size"),
    save_full_frames: bool = typer.Option(False, help="save full frames"),
    save_flatten: bool = typer.Option(False, help="save flatten"),
):
    annotation_paths = list(annotation_path.rglob("*.xml"))
    output_path.mkdir(exist_ok=True, parents=True)
    pbar = tqdm(natsorted(annotation_paths))
    for annotation_path in pbar:
        dataset_name = annotation_path.parent.parent.name
        pbar.set_description(dataset_name)
        cut_rider_from_idd_xml(
            annotation_path, img_path, output_path, pad_size, save_full_frames, save_flatten
        )


if __name__ == "__main__":
    app()
