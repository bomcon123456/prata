from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import typer
import shutil

app = typer.Typer(pretty_exceptions_show_locals=True)


@app.command()
def main(
    pose_dir: Path = typer.Argument(..., help="Dir consists of csvs merged everything"),
    videoid_dir: Path = typer.Argument(..., help="Dir consists of csvs has video_id"),
    output_dir: Path = typer.Argument(..., help="output dir"),
):
    pose_csvs = list(pose_dir.rglob("*.csv"))
    for pose_csv in tqdm(pose_csvs):
        relative_path = pose_csv.relative_to(pose_dir)
        videoid_path = videoid_dir / relative_path
        df = pd.read_csv(pose_csv)
        videoid_df = pd.read_csv(videoid_path)
        assert len(df) == len(videoid_df)
        for i, (row_left, row_right) in enumerate(
            zip(df.itertuples(), videoid_df.itertuples())
        ):
            assert row_left.frameid == row_right.frameid
            row_index = df.index[i]
            df.loc[row_index, "video_id"] = row_right.video_id
            df.loc[row_index, "user_id"] = row_right.user_id
        output_path = output_dir / relative_path
        output_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_path.as_posix(), index=False)


if __name__ == "__main__":
    app()
