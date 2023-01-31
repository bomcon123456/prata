import copy
import json
import numpy as np
import xmltodict
from pathlib import Path
import cv2
from rich import print

from prata.common.boxes import ious, iou, is_in_box


def get_rider_and_bike(objects):
    res = []
    riders = []
    bikes = []
    bicycles = []
    l = {"rider": riders, "motorcycle": bikes, "bicycle": bicycles}
    for obj in objects:
        if obj["name"] in ["rider", "motorcycle", "bicycle"]:
            res.append(obj)
            l[obj["name"]].append(obj)
    return res, l


def associate_riders_bikes(objects_dict):
    v_result = copy.deepcopy(objects_dict["motorcycle"]) + copy.deepcopy(
        objects_dict["bicycle"]
    )
    p_result = copy.deepcopy(objects_dict["rider"])
    for person in p_result:
        p_ltrb = get_ltrb(person)
        v_ltrbs = np.array(list(map(get_ltrb, v_result)))
        ious_ = ious([p_ltrb], v_ltrbs)
        max_iou, max_iou_idx = np.max(ious_, axis=1), np.argmax(ious_, axis=1)
        if max_iou > 0.05:
            linked_bike = v_result[max_iou_idx[0]]
            person["linked_bike"] = copy.deepcopy(linked_bike)
            if "linked_ppl" in person["linked_bike"]:
                del person["linked_bike"]["linked_ppl"]
            person["linked_bike"]["iou"] = max_iou[0]
            if "linked_ppl" not in linked_bike:
                linked_bike["linked_ppl"] = []
            p_ = copy.deepcopy(person)
            p_["iou"] = max_iou[0]
            if "linked_bike" in p_:
                del p_["linked_bike"]
            linked_bike["linked_ppl"].append(p_)

    for vehicle in v_result:
        if "linked_ppl" in vehicle and len(vehicle["linked_ppl"]) > 0:
            v_ltrb = get_ltrb(vehicle)
            p_ltrbs = np.array(list(map(get_ltrb, vehicle["linked_ppl"])))
            x1 = min((v_ltrb[0], *p_ltrbs[:, 0].tolist()))
            y1 = min((v_ltrb[1], *p_ltrbs[:, 1].tolist()))
            x2 = max((v_ltrb[2], *p_ltrbs[:, 2].tolist()))
            y2 = max((v_ltrb[3], *p_ltrbs[:, 3].tolist()))
            vehicle["big_crop"] = [x1, y1, x2, y2]
    return p_result, v_result


def get_ltrb(obj):
    bbox_obj = obj["bndbox"]
    ltrb = [bbox_obj["xmin"], bbox_obj["ymin"], bbox_obj["xmax"], bbox_obj["ymax"]]
    return np.array(ltrb, dtype=np.int32)


def cut_rider_from_idd_xml(
    xml_path: Path,
    img_basepath: Path,
    out_basepath: Path,
    pad_size=15,
    save_fullframe=False,
    save_flatten=False,
):
    with open(xml_path, "r") as xml_file:
        data_dict = xmltodict.parse(xml_file.read())["annotation"]
    img_path = (
        img_basepath
        / xml_path.parent.parent.name
        / xml_path.parent.name
        / (xml_path.stem + ".jpg")
    )
    basename = f"{xml_path.parent.parent.name}_{xml_path.parent.name}_{xml_path.stem}"
    if save_flatten:
        out_basepath = out_basepath
    else:
        out_basepath = out_basepath / xml_path.parent.parent.name / xml_path.parent.name
    out_basepath.mkdir(exist_ok=True, parents=True)
    assert img_basepath.exists()
    # print(data_dict)
    if "object" not in data_dict or not isinstance(data_dict["object"], list):
        # print("No object")
        return
    objs, objs_dict = get_rider_and_bike(data_dict["object"])
    if len(objs) == 0:
        # print("No riders/bikes")
        return
    if (
        (len(objs_dict["rider"]) == 0)
        or (len(objs_dict["motorcycle"]) == 0)
        or (len(objs_dict["bicycle"]) == 0)
    ):
        return

    _, vehicles = associate_riders_bikes(objs_dict)

    img = cv2.imread(img_path.as_posix())

    if save_fullframe:
        fullframe_path = out_basepath / "fullframes"
        fullframe_path.mkdir(exist_ok=True, parents=True)
        viz = img.copy()
        for obj in objs:
            color = (255, 0, 0) if obj["name"] == "rider" else (0, 0, 255)
            ltrb = get_ltrb(obj)
            cv2.rectangle(viz, (ltrb[:2]), (ltrb[2:]), color, 2, 2)
        cv2.imwrite((fullframe_path / f"{basename}.png").as_posix(), viz)

    fullframe_path = out_basepath / "fullframes"
    for i, v in enumerate(vehicles):
        if "big_crop" not in v:
            continue
        l, t, r, b = v["big_crop"]
        w,h = r-l, b-t
        if w < 60 or h < 60:
            continue
        l = max(l - pad_size, 0)
        t = max(t - pad_size, 0)
        r = min(r + pad_size, img.shape[1])
        b = min(b + pad_size, img.shape[0])
        cropped = img[t:b, l:r]
        if save_flatten:
            cur_out = out_basepath / f"{v['name']}/{len(v['linked_ppl'])}"
            name = f"{basename}_{str(i) + '.jpg'}"
        else:
            cur_out = out_basepath / f"{v['name']}/{len(v['linked_ppl'])}"
            name = f"{i}.jpg"
        cur_out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            (cur_out / name).as_posix(),
            cropped,
        )


if __name__ == "__main__":
    xml_path = Path(
        "/home/ubuntu/workspace/trungdt21/kidnap/data/idd/IDD_Detection/Annotations/frontFar/BLR-2018-03-22_17-39-26_2_frontFar/000186_r.xml"
    )
    big_cat_name = xml_path.parent.parent.name
    img_basepath = Path(
        "/home/ubuntu/workspace/trungdt21/kidnap/data/idd/IDD_Detection/JPEGImages"
    )
    cut_rider_from_idd_xml(
        xml_path, img_basepath, Path("./tmp/rider"), save_fullframe=True
    )
