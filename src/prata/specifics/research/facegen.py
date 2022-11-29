import json
import os
import typer
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil
import matplotlib.pyplot as plt


app = typer.Typer()

@app.command()
def ffhq_poseify(
    input_csv: Path = typer.Argument(...,help="input path", exists=True, file_okay=True),
    plot_dist: bool = typer.Option(True, help="plot pose dist"),
    output_path: Path = typer.Option(".", help="output path"),
    prefix: str = typer.Option("", help="prefix to name")
):
    df = pd.read_csv(input_csv)
    output_path.mkdir(parents=True, exist_ok=True)
    yaws = df["head_yaw"].to_list()
    pitchs = df["head_pitch"].to_list()
    if plot_dist:
        plt.hist(yaws, bins=range(-90,90,10), alpha=0.5, label="Yaw")
        plt.hist(pitchs, bins=range(-90,90,10), alpha=0.5, label="Pitch")
        plt.title("FFHQ's Pitch, Yaw distribution")
        plt.savefig((output_path/"pitch_yaw.png").as_posix())
    labels = [] 
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_name = prefix + str(row.image_number).zfill(5) + ".png"
        label = "frontal"
        if row.head_yaw > 45:
            label = "profile_left"
        elif row.head_yaw < -45:
            label = "profile_right"
        elif row.head_pitch > 30:
            label = "profile_up"
        elif row.head_pitch < -30:
            label = "profile_down"
        pair = [img_name, label]
        labels.append(pair)
    out = {"labels":labels}
    with open(output_path/"datasets.json", "w") as f:
        json.dump(out, f)
        
@app.command()
def selfgen_poseify(
    input_basepath: Path = typer.Argument(...,help="input path", exists=True, dir_okay=True),
    output_path: Path = typer.Option(".", help="output path"),
    prefix: str = typer.Option("", help="prefix to name")
):
    txt_files = input_basepath.rglob("*/info.txt")
    labels = []
    for i, txt_file in enumerate(txt_files):
        if (i%2000 == 0):
            print(f"Process {i} files...")
        rela_path = txt_file.relative_to(input_basepath)
        with open(txt_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2
        line = lines[-1].strip().split()
        filename = line[0]
        filepath = rela_path.parent / f"{filename}.png"
        assert (input_basepath / filepath).exists(), f"{filepath} not exists."
        pitch = float(line[-3])
        yaw = float(line[-2])
        label = "frontal"
        if yaw > 45:
            label = "profile_left"
        elif yaw < -45:
            label = "profile_right"
        elif pitch > 30:
            label = "profile_up"
        elif pitch < -30:
            label = "profile_down"
        pair = [filepath.as_posix(), label]
        labels.append(pair)
    
    out = {"labels":labels}
    with open(output_path/"datasets.json", "w") as f:
        json.dump(out, f)
            
    pass
    

if __name__ == "__main__":
    app()