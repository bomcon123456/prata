from pathlib import Path
import json
from tqdm.rich import tqdm
import pandas as pd

df = pd.read_csv("/lustre/scratch/client/guardpro/trungdt21/research/face_gen/data/FFHQ/ffhqpose.csv")

basepath = Path("compose")
img_paths = list(basepath.rglob("*.png"))
res = {"labels": []}
for img_path in tqdm(img_paths):
    if img_path.parent.name == "ffhq1024":
        img_number = int(img_path.stem)
        row = df[df["image_number"] == img_number].iloc[0]
        p,r,y = row["head_pitch"],row["head_roll"],row["head_yaw"]
        if abs(p) > 35 or abs(r) > 30 or abs(y) > 40:
            label = "extreme"
        else:
            label = "frontal"
    else:
        label="extreme"
    labelstr_to_idx = {"frontal": 0, "extreme": 1}
    label = int(labelstr_to_idx[label])
    res["labels"].append((img_path.relative_to(basepath).as_posix(), label))
with open("./compose/dataset.json", "w") as f:
    json.dump(res, f, indent=2)
