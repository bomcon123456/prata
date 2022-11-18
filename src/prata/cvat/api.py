import typer
from pathlib import Path
from .utils import create_tasks


def push_task(
    project_id: int = typer.Argument(..., help="Project id"),
    datapath: Path = typer.Argument(..., help="datapath", exists=True),
    annotation_path: Path = typer.Option(None, help="annotation_path",exists=True),
    annotation_format: str = typer.Option("COCO%201.0", help="Annotation format"),
    subset_name: str = typer.Option(None, help="Subset name"),
    task_name_with_parent: bool = typer.Option(False, help="Task name will be concatenation of parent folder + filename")
):
    tmp = datapath
    zip_files = list(tmp.glob("**/*.zip"))

    create_tasks(project_id, zip_files, annotation_path, annotation_format, subset_name, task_name_with_parent)
