import typer
import cv2
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm


app = typer.Typer()


def f(img_path):
    if "mirror" not in img_path.stem:
        out_path = img_path.parent / f"{img_path.stem}_mirror{img_path.suffix}"
    else:
        out_path = (
            img_path.parent / f"{img_path.stem.replace('_mirror','')}{img_path.suffix}"
        )
    if out_path.exists():
        return
    img = cv2.imread(img_path.as_posix())
    flipped = cv2.flip(img, 1)
    cv2.imwrite(out_path.as_posix(), flipped)


@app.command()
def main(
    img_dir: Path = typer.Argument(..., exists=True, dir_okay=True),
    nprocs: int = typer.Option(16, help="Nprocs"),
):
    img_paths = img_dir.rglob("*.png")
    img_paths = list(filter(lambda x: "mirror" not in x.as_posix(), img_paths))

    with Pool(nprocs) as p:
        list(
            tqdm(
                p.imap(
                    f,
                    img_paths,
                ),
                total=len(img_paths),
            )
        )


@app.command()
def gen_missing(
    img_dir: Path = typer.Argument(..., exists=True, dir_okay=True),
    nprocs: int = typer.Option(16, help="Nprocs"),
):
    img_paths = list(img_dir.rglob("*.png"))

    with Pool(nprocs) as p:
        list(
            tqdm(
                p.imap(
                    f,
                    img_paths,
                ),
                total=len(img_paths),
            )
        )


if __name__ == "__main__":
    app()
