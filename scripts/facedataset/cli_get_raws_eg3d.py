from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import typer
import shutil

app = typer.Typer(pretty_exceptions_show_locals=True)


@app.command()
def celebvhq(
    metadata_path: Path = typer.Argument(..., help="CelebVHQ metadata path"),
    raw_img_dir: Path = typer.Argument(..., help="Original Raw img path"),
    output_dir: Path = typer.Argument(..., help="Original Raw img path"),
):
    with open(metadata_path, "r") as f:
        lookup = json.load(f)
    align_base_dir = Path(
        "/lustre/scratch/client/scratch/research/group/anhgroup/ducvh5/CelebV-HQ_v2/processed/extracted_cropped_face_results_ffhq"
    )
    for curfilename, d in tqdm(lookup.items(), total=len(lookup)):
        aligned_path = Path(d["original_path"])
        relative_path = aligned_path.relative_to(align_base_dir)
        raw_parent_path = raw_img_dir / relative_path.parent
        raw_img_path = raw_img_dir / relative_path.name
        if not raw_img_path.exists():
            strips = relative_path.stem.split("_")
            assert len(strips) == 2
            filename = strips[0]
            raw_img_path = raw_parent_path / f"{filename}.png"
            assert raw_img_path.exists()
        dest_path = output_dir / curfilename
        shutil.copy(raw_img_path, dest_path)


@app.command()
def vfhq(
    aligned_dir: Path = typer.Argument(..., help="Aligned path"),
    test_parquet_path: Path = typer.Argument(..., help="parquet path"),
    train_parquet_path: Path = typer.Argument(..., help="parquet path"),
    raw_img_dir: Path = typer.Argument(..., help="Original Raw img path"),
    output_dir: Path = typer.Argument(..., help="Original Raw img path"),
):
    test_df = pd.read_parquet(test_parquet_path)
    train_df = pd.read_parquet(train_parquet_path)
    image_paths = list(aligned_dir.glob("*.png"))
    counter = 0
    for image_path in tqdm(image_paths):
        splits = image_path.stem.split("_")

        for i, split in enumerate(splits):
            if "ypr" in split:
                img_name_start_idx = i - 1
                break
        user_id = "_".join(splits[:img_name_start_idx])
        aligned_name = "_".join(splits[img_name_start_idx:-1])
        if not splits[-1].isdigit():
            aligned_name += f"_{splits[-1]}"
        aligned_path = f"{user_id}/{aligned_name}.png"
        is_test = True
        row = test_df[test_df["aligned_path"] == aligned_path]
        if len(row) == 0:
            row = train_df[train_df["aligned_path"] == aligned_path]
            is_test = False
        row = row.iloc[0]
        frameidx = row["frameid"]
        raw_filename = str(frameidx).zfill(8) + ".png"
        raw_filepath = raw_img_dir / ("test" if is_test else "train") / user_id
        raw_filepath = list(raw_filepath.rglob(raw_filename))
        #assert len(raw_filepath) == 1, f"exists {len(raw_filepath)} files: {raw_filepath}"
        if len(raw_filepath) > 1:
            counter +=1
        for i, p in enumerate(raw_filepath):
            out_path = output_dir / (image_path.stem + f"_{i}" + image_path.suffix)
            shutil.copy(p, out_path)
    print(f"total duplicates: {counter}")
    # align_base_dir = Path("/lustre/scratch/client/scratch/research/group/anhgroup/ducvh5/CelebV-HQ_v2/processed/extracted_cropped_face_results_ffhq")
    # for curfilename, d in tqdm(lookup.items(), total=len(lookup)):
    #     aligned_path = Path(d["original_path"])
    #     relative_path = aligned_path.relative_to(align_base_dir)
    #     raw_parent_path = raw_img_dir / relative_path.parent
    #     raw_img_path = raw_img_dir / relative_path.name
    #     if not raw_img_path.exists():
    #         strips = relative_path.stem.split("_")
    #         assert len(strips) == 2
    #         filename = strips[0]
    #         raw_img_path = raw_parent_path / f"{filename}.png"
    #         assert raw_img_path.exists()
    #     dest_path = output_dir / curfilename
    #     shutil.copy(raw_img_path, dest_path)


if __name__ == "__main__":
    app()
