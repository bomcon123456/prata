import streamlit as st
import base64
from io import BytesIO
from pathlib import Path
import pandas as pd
from PIL import Image
import os
from st_clickable_images import clickable_images
import typer
import zipfile
import base64
from natsort import natsorted

CACHE_PATH = Path("./tmp/posegrid")

app = typer.Typer()


@st.cache_data
def globs(basepath: Path, pattern: str):
    files = natsorted(list(basepath.rglob(pattern)))

    return files


def bin_to_color(bin):
    d = {
        "frontal": "white",
        "profile_left": "red",
        "profile_right": "green",
        "profile_up": "yellow",
        "profile_down": "blue",
        "profile_extreme": "purple",
    }
    return d[bin]


@st.cache_data
def read_img_from_zip(zip_path: Path):
    res = {}
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for file_name in zip_file.namelist():
            if not file_name.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".gif")
            ):
                continue
            with zip_file.open(file_name) as my_file:
                image_bytes = my_file.read()
                with Image.open(BytesIO(image_bytes)) as img:
                    img_resized = img.resize((50, 50))
                    img_resized_bytes = BytesIO()
                    img_resized.save(img_resized_bytes, format="JPEG")
                    img_base64 = base64.b64encode(img_resized_bytes.getvalue()).decode(
                        "utf-8"
                    )
                res[Path(file_name).stem] = img_base64
    return res


if "csv_counter" not in st.session_state:
    st.session_state.csv_counter = 0


@app.command()
def main(
    csv_paths: Path = typer.Argument(..., help="csv path", exists=True, dir_okay=True),
    zip_paths: Path = typer.Argument(..., help="zip path", exists=True, dir_okay=True),
    posebin: str = typer.Option("softbin", help="column name for posebin"),
):
    st.title("Pose Grid")

    csvs = globs(csv_paths, "*.csv")

    with st.sidebar:
        filter_box = st.sidebar.selectbox(
            "Filter",
            (
                "all",
                "profile_horizontal",
                "profile_vertical",
                "frontal",
                "profile_extreme",
            ),
        )
        current_csv = csvs[st.session_state.csv_counter]
        st.title(f"ID: {current_csv.stem}")
        img_size = st.slider("Image Size", 50, 100, 50, step=5, key="img_size")
        if st.button("Next"):
            st.session_state.csv_counter = min(
                st.session_state.csv_counter + 1, len(csvs) - 1
            )
        if st.button("Prev"):
            st.session_state.csv_counter = max(st.session_state.csv_counter - 1, 0)
        if st.button("Find first have image"):
            while st.session_state.csv_counter < len(csvs) - 1:
                current_csv = csvs[st.session_state.csv_counter]
                df = st.session_state.df
                filtered_df = (
                    df[df[posebin] == filter_box] if filter_box != "all" else df
                )
                if len(filtered_df) == 0:
                    st.session_state.csv_counter = min(
                        st.session_state.csv_counter + 1, len(csvs) - 1
                    )
                else:
                    break
            else:
                st.text(f"All ids don't have {filter_box} bin")

    current_csv = csvs[st.session_state.csv_counter]

    if "df" not in st.session_state:
        df = pd.read_csv(current_csv)
        st.session_state.df = df

    df = st.session_state.df
    current_zip = zip_paths / f"{current_csv.stem}.zip"

    filtered_df = df[df[posebin] == filter_box] if filter_box != "all" else df
    if len(filtered_df) == 0:
        st.text("No images")
    else:
        images_dict = read_img_from_zip(current_zip)
        dict_df = filtered_df.to_dict("records")
        images = []
        colors = []
        for d in dict_df:
            fid = str(d["frameid"]).zfill(8)
            images.append(images_dict[fid])
            if filter_box == "profile_horizontal":
                if d["hardbin"] in ["profile_left", "profile_right"]:
                    bin = d["hardbin"]
                elif d["mhp_yaw"] is not None:
                    bin = "profile_right" if d["mhp_yaw"] > 0 else "profile_left"
                elif d["synergy_yaw"] is not None:
                    bin = "profile_right" if d["mhp_yaw"] > 0 else "profile_left"
            elif filter_box == "profile_vertical":
                if d["hardbin"] in ["profile_up", "profile_down"]:
                    bin = d["hardbin"]
                elif d["mhp_pitch"] is not None:
                    bin = "profile_up" if d["mhp_pitch"] > 0 else "profile_down"
                elif d["synergy_pitch"] is not None:
                    bin = "profile_up" if d["synergy_pitch"] > 0 else "profile_down"
            else:
                bin = d[posebin]
            color = bin_to_color(bin)
            colors.append(color)

        clicked = show_grid_of_images(images, colors, img_size)


# Load the images and display them in a grid
def show_grid_of_images(image_files, colors, img_size):
    images = []
    for file in image_files:
        images.append(f"data:image/jpeg;base64,{file}")

    clicked = clickable_images(
        images,
        titles=[i for i in range(len(image_files))],
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style=[
            {
                "margin": "1px",
                "height": f"{img_size}px",
                "width": f"{img_size}px",
                "border": f"3px solid {colors[i]}",
            }
            for i in range(len(image_files))
        ],
    )
    return clicked


if __name__ == "__main__":
    app(standalone_mode=False)
