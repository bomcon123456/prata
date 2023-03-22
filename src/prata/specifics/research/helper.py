from pathlib import Path
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious
import pandas as pd

__all__= ["ious", "read_csv_from_txt", "merge3posedf"]

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float),
    )

    return ious


def read_csv_from_txt(txt_path: Path, skiprows=7,skipfooter=1):
    df = pd.read_csv(txt_path.as_posix(), skiprows=skiprows, delim_whitespace=True, header=None, skipfooter=skipfooter)
    colnames16 = ["frameid", "idx", "x1", "y1", "x2", "y2"] + [f"lmk{i//2}_{'x' if i%2==0 else 'y'}" for i in range(10)]
    colnames19 = colnames16 + ["synergy_head_pitch", "synergy_head_yaw", "synergy_head_roll"]
    colnames223 = colnames16.copy() + ["poseanh_head_pitch", "poseanh_head_yaw", "poseanh_head_roll"]
    for i in range(68):
        colnames223.extend([f"3dlmk{i}_x",f"3dlmk{i}_y",f"3dlmk{i}_z"])
    if len(df.columns) == 16:
        finalcol = colnames16
    elif len(df.columns) == 19:
        finalcol = colnames19
    elif len(df.columns) == 223:
        finalcol = colnames223
    df.rename(columns=dict(zip(df.columns, finalcol)), inplace=True)
    return df

def merge3posedf(vfhq, synergy, poseanh):
    vfhq = read_csv_from_txt(Path("/home/termanteus/workspace/common/data_scripts/vfhqannot.txt"))
    poseanh = read_csv_from_txt(Path("/home/termanteus/workspace/common/data_scripts/poseanh.txt"))
    synergy = read_csv_from_txt(Path("/home/termanteus/workspace/common/data_scripts/synergy.txt"))
    merged = pd.merge(vfhq,poseanh, on=['frameid', 'idx', 'x1', 'x2', 'y1', 'y2', *[f'lmk{i}_x' for i in range(5)], *[f'lmk{i}_y' for i in range(5)]], how='inner')
    merged = pd.merge(synergy,merged, on=['frameid', 'idx', 'x1', 'x2', 'y1', 'y2', *[f'lmk{i}_x' for i in range(5)], *[f'lmk{i}_y' for i in range(5)]], how='inner')
    return merged
    

if __name__ == "__main__":

    vfhq = read_csv_from_txt(Path("/home/termanteus/workspace/common/data_scripts/vfhqannot.txt"))
    poseanh = read_csv_from_txt(Path("/home/termanteus/workspace/common/data_scripts/poseanh.txt"))
    synergy = read_csv_from_txt(Path("/home/termanteus/workspace/common/data_scripts/synergy.txt"))
    print(merge3posedf(vfhq, synergy, poseanh).head())
    # print(df3.head())
