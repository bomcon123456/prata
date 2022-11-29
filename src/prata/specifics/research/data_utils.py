from pathlib import Path
import shutil
from tqdm import tqdm

def oneclassify_onefolder(input_path:Path, output_path: Path):
    files = list(input_path.rglob("*"))
    for file in tqdm(files):
        out_path = output_path / (file.relative_to(input_path))
        out_path.parent.mkdir(exist_ok=True, parents=True)
        if file.suffix == ".txt":
            fout = open(out_path, "w")
            fin = open(file, "r")
            lines = fin.readlines()
            lines = list(map(lambda x: x.strip().split(), lines))
            # collapse label_id to 0 (one class)
            for line in lines:
                line[0] = "0"
                for number in line[1:]:
                    if float(number) > 1.:
                        number = "1."
            lines = list(map(" ".join,lines))
            lines = list(map(lambda x: x+"\n",lines))
            fout.writelines(lines)
        else:
            shutil.copy2(file, out_path)