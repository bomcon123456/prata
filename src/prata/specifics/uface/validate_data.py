from genericpath import exists
import typer
import pandas as pd
from tqdm.rich import tqdm
from pathlib import Path
import shutil

app = typer.Typer()

@app.command()
def save_to_enroll_verify(
    csv_path: Path = typer.Argument(...,help="csv path", exists=True, file_okay=True),
    img_path: Path = typer.Argument(...,help="img base path", exists=True, dir_okay=True),
    output_path: Path = typer.Argument(..., help="csv path"),
):
    df = pd.read_csv(csv_path)
    for row in tqdm(df.itertuples(), total=len(df)):
        id_ = row.id
        type = row.type
        verify_type = row.verify_type
        path = Path(row.cropped_img_path)
        outp = output_path / verify_type / id_ / type / path.name
        outp.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2((img_path/path), outp)

if __name__ == "__main__":
    app()