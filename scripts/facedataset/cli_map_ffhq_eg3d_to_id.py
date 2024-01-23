import typer
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from scipy.io import loadmat

app = typer.Typer()


@app.command()
def main(
    ffhq_eg3d_json_path: Path = typer.Argument(
        ..., help="Json path of ffhq eg3d", exists=True, file_okay=True
    ),
    ffhq_eg3d_mat_path: Path = typer.Argument(
        ..., help="mat path of ffhq eg3d", exists=True, dir_okay=True
    ),
    outdir: Path = typer.Argument(..., help="outdir", dir_okay=True),
):
    with open(ffhq_eg3d_json_path, "r") as f:
        labels = json.load(f)["labels"]
    image_names = sorted([label[0] for label in labels])
    lookup = {}
    result = {"camera_angles": {}}
    result2 = {"camera_angles": {}}
    failed = set()
    for idx, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        idx_str = f"{idx:08d}"
        archive_fname = f"{idx_str[:5]}/img{idx_str}.png"
        lookup[archive_fname] = image_name

        # calculate pose
        is_mirror = "mirror" in image_name
        real_image_name = image_name.replace("_mirror", "")
        mat_path = (ffhq_eg3d_mat_path / real_image_name).with_suffix(".mat")
        if not mat_path.exists():
            failed.add(mat_path.as_posix())
            continue
        pitch, yaw, roll = loadmat(mat_path.as_posix())["angle"][0]
        pitch_degree = pitch / np.pi * 180
        pitch_degree *= -1
        pitch_degree_offset = pitch_degree + 90
        pitch_radian_offset = pitch_degree_offset * (np.pi / 180)
        if is_mirror:
            yaw = -yaw
        new_angles = [float(yaw), float(pitch_radian_offset), 0.0]
        result["camera_angles"][archive_fname] = new_angles
        result2["camera_angles"][image_name] = new_angles

    outdir.mkdir(exist_ok=True, parents=True)
    lookup_path = outdir / "lookup.json"
    dataset_path = outdir / "dataset.json"
    dataset_path2 = outdir / "dataset_real_id.json"
    failed_path = outdir / "failed.txt"
    with open(lookup_path, "w") as f:
        json.dump(lookup, f, indent=2)
    with open(dataset_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(dataset_path2, "w") as f:
        json.dump(result2, f, indent=2)
    with open(failed_path, "w") as f:
        f.write("\n".join(failed))


if __name__ == "__main__":
    app()
