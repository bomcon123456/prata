import typer
from functools import partial
import json
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from deepface import DeepFace


app = typer.Typer()


def f(img_path, img_dir, outdir):
    json_path = outdir / img_path.relative_to(img_dir).with_suffix(".json")
    if json_path.exists():
        return
    objs = DeepFace.analyze(
        img_path=img_path.as_posix(),
        actions=["age", "gender", "race", "emotion"],
        enforce_detection=False,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path.as_posix(), "w") as f:
        json.dump(objs, f)


@app.command()
def main(
    img_dir: Path = typer.Argument(..., exists=True, dir_okay=True),
    outdir: Path = typer.Argument(..., dir_okay=True),
    nprocs: int = typer.Option(4, help="Nprocs"),
):
    img_paths = img_dir.rglob("*.png")
    img_paths = list(filter(lambda x: "mirror" not in x.as_posix(), img_paths))

    for img_path in tqdm(img_paths):
        f(img_path, img_dir, outdir)
    # with Pool(nprocs) as p:
    #     list(
    #         tqdm(
    #             p.imap(
    #                 partial(
    #                     f,
    #                     outdir=outdir,
    #                     img_dir=img_dir,
    #                 ),
    #                 img_paths,
    #             ),
    #             total=len(img_paths),
    #         )
    #     )


if __name__ == "__main__":
    app()
