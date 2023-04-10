from pathlib import Path

import fiftyone as fo
import typer

app = typer.Typer()


@app.command()
def create_coco_dataset(
    name: str = typer.Argument(..., help="name"),
    data_path: Path = typer.Argument(..., help="path to dataset", exists=True),
    label_path: Path = typer.Option(None, help="path to dataset"),
):
    d = dict(dataset_type=fo.types.COCODetectionDataset, name=name)
    if label_path is None:
        d["dataset_dir"] = data_path.resolve().as_posix()
    else:
        d["data_path"] = data_path.resolve().as_posix()
        d["labels_path"] = label_path.resolve().as_posix()

    dataset = fo.Dataset.from_dir(**d)
    session = fo.launch_app(dataset)
    input()
