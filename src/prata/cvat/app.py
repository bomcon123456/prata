import typer
from pathlib import Path

from . import converter, api

app = typer.Typer()

@app.command()
def datumaro_to_coco(
    input_path: Path = typer.Argument(..., help="Path to input", exists=True),
    output_path: Path = typer.Argument(..., help="Path to output"),
    save_images: bool = typer.Option(False, help="save image too"),
    filter_negative: bool = typer.Option(False, help="filter negative"),
    debug: bool = typer.Option(False, help="debug")
):
    converter.datumaro_to_coco(input_path, output_path, save_images, filter_negative, debug)

@app.command()
def push_task(
    project_id: int = typer.Argument(..., help="Project id"),
    datapath: Path = typer.Argument(..., help="datapath", exists=True),
    annotation_path: Path = typer.Option(None, help="annotation_path",exists=True),
    annotation_format: str = typer.Option("COCO%201.0", help="Annotation format"),
    subset_name: str = typer.Option(None, help="Subset name"),
    task_name_with_parent: bool = typer.Option(False, help="Task name will be concatenation of parent folder + filename")
):
    api.push_task(project_id, datapath, annotation_path, annotation_format, subset_name, task_name_with_parent)

if __name__ == "__main__":
    app()