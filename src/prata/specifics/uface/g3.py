import typer
from pathlib import Path
import json
import pandas as pd

app = typer.Typer()


def create_caminfo(name, ip, fps=10, event_type="need_to_monitor", rois=list()):
    return {
        "name": name,
        "url": f"rtsp://admin:123456aA@{ip}:554/Streaming/Channels/101",
        "urlNVR": "rtsp://admin:qksYnVL5Su@172.22.25.26:1200/Streaming/Channels/2101",
        "cam_url": f"rtsp://admin:123456aA@{ip}:554/Streaming/Channels/101",
        "fps": fps,
        "cam_name": f"BLACKLIST@{name}",
        "cam_id": f"BLACKLIST@{name}",
        "event_type": event_type,
        "rois": rois,
    }


@app.command()
def create_json(
    csv_path: Path = typer.Argument(..., help="csv path"),
    json_path: Path = typer.Argument(..., help="json path"),
    output_path: Path = typer.Argument(..., help="output path"),
    override_cam_info: bool = typer.Option(False, help="remove previous cam info"),
    event_type: str = typer.Option("need_to_monitor", help="Event type"),
    fps: int = typer.Option(10, help="fps"),
):
    with open(json_path, "r") as f:
        obj = json.load(f)
    df = pd.read_csv(csv_path)
    new_cam_infos = []
    for row in df.itertuples():
        name = row.name
        ip = row.ip
        caminfo = create_caminfo(name, ip, fps, event_type)
        new_cam_infos.append(caminfo)
    if override_cam_info:
        obj["cam_infor"] = new_cam_infos
    else:
        obj["cam_infor"] += new_cam_infos
    with open(output_path, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    app()
