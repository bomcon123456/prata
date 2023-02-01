import yaml
import json

from yaml.loader import SafeLoader
from typing import List, Optional, Dict

from pathlib import Path

from pydantic import BaseModel
import numpy as np

ANNO_TYPES = ["activities", "geom", "types"]


def combine_tlbr_upperbound(tlbrs: np.ndarray):
    max_tlbr = np.max(tlbrs, axis=0)
    min_tlbr = np.min(tlbrs, axis=0)
    res = max_tlbr.copy()
    res[:2] = min_tlbr[:2]
    return res


def expand_tlbr(tlbr: np.ndarray, factor: float, hw: np.ndarray):
    t, l, b, r = tlbr
    t = float(t - factor * t)
    l = float(l - factor * l)
    r = float(r + factor * r)
    b = float(b + factor * b)
    t = max(0, t)
    l = max(0, l)
    r = min(hw[1], r)
    b = min(hw[0], b)
    return np.array((t, l, b, r), np.float32)


class LocTimestamp(BaseModel):
    timestamp: int
    timestamp_sec: Optional[float] = None
    tlbr: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @property
    def tlwh(self):
        tlwh = self.tlbr.copy()
        tlwh[2:] -= tlwh[:2]
        return tlwh

    @staticmethod
    def parse_from_yaml(d):
        ts = d["ts0"]
        ts_sec = getattr(d, "ts1", None)
        l, t, r, b = np.array(list(map(float, d["g0"].split(" "))))

        return LocTimestamp(timestamp=ts, timestamp_sec=ts_sec, tlbr=np.array((t, l, b, r)))


class Actor(BaseModel):
    id: int
    timespan: List[int]
    type: Optional[str] = None
    loc: Optional[LocTimestamp] = []

    def get_bounded_tlbr(self):
        tlbrs = np.array([l.tlbr for l in self.loc])
        return combine_tlbr_upperbound(tlbrs)


class Activity(BaseModel):
    id: int
    class_name: str
    timespan: List[int]
    actors: List[Actor]

    def get_bounded_tlbr(self, factor=None, hw=None):
        tlbrs = []
        for actor in self.actors:
            tlbr = actor.get_bounded_tlbr()
            tlbrs.append(tlbr)
        tlbrs = np.array(tlbrs)
        tlbr = combine_tlbr_upperbound(tlbrs)
        if factor is not None and hw is not None:
            tlbr = expand_tlbr(tlbr, factor, hw)
        return tlbr


def parse_activities_path(yaml_path: Path):
    with open(yaml_path, "r") as f:
        activities = yaml.load(f, Loader=SafeLoader)
    activities_res = []
    for obj_ in activities:
        if "act" not in obj_:
            continue
        obj = obj_["act"]
        try:
            activity_name = obj["act3"]
        except:
            activity_name = obj["act2"]
        assert (
            len(list(activity_name.keys())) == 1
        ), f"Activities keys has more than 1: {activity_name.keys()}"
        activity_name = list(activity_name.keys())[0]
        id = obj["id2"]
        timespan = obj["timespan"]
        assert len(timespan) == 1, f"{timespan}: has at least 2 samples"
        timespan = timespan[0]["tsr0"]
        actors_ = obj["actors"]
        actors = []
        for actor_ in actors_:
            actor_timespan = actor_["timespan"]
            assert len(actor_timespan) == 1, f"{actor_timespan}: has at least 2 samples"
            actor_timespan = actor_timespan[0]["tsr0"]
            actor = Actor(id=actor_["id1"], timespan=actor_timespan)
            actors.append(actor)

        activity = Activity(
            id=id, timespan=timespan, class_name=activity_name, actors=actors
        )
        activities_res.append(activity)
    return activities_res


def parse_geom_path(yaml_path: Path, actorid_to_actor: Dict):
    with open(yaml_path, "r") as f:
        actors = yaml.load(f, Loader=SafeLoader)
    for obj_ in actors:
        if "geom" not in obj_:
            continue
        obj = obj_["geom"]
        actor_id = obj["id1"]
        actor = actorid_to_actor[actor_id]
        lt = LocTimestamp.parse_from_yaml(obj)
        actor.loc.append(lt)
    return actorid_to_actor


def parse_types_path(yaml_path: Path, actorid_to_actor: Dict):
    with open(yaml_path, "r") as f:
        types = yaml.load(f, Loader=SafeLoader)
    for obj_ in types:
        if "types" not in obj_:
            continue
        obj = obj_["types"]
        actor_id = obj["id1"]
        tmp = list(obj["cset3"].keys())
        assert len(tmp) == 1, f"There are multiple types for this actor: {obj}"
        actor_type = tmp[0]
        actorid_to_actor[actor_id].type = actor_type


def get_all_yaml_from_one(yaml_path: Path):
    filename = yaml_path.stem
    anno_type = filename.split(".")[-1]
    assert anno_type in ANNO_TYPES

    filename_raw = ".".join(filename.split(".")[:-1])
    activities_path = yaml_path.parent / f"{filename_raw}.activities.yml"
    geom_path = yaml_path.parent / f"{filename_raw}.geom.yml"
    types_path = yaml_path.parent / f"{filename_raw}.types.yml"

    assert activities_path.exists()
    assert geom_path.exists()
    assert types_path.exists()

    return activities_path, geom_path, types_path


def parse_yaml(yaml_path: Path):
    activities_path, geom_path, types_path = get_all_yaml_from_one(yaml_path)
    activities = parse_activities_path(activities_path)

    actors_ = {}
    for activity in activities:
        for actor in activity.actors:
            if actor.id in actors_:
                print(f"Actor_id {actor.id} existed in dict: {actor} vs {actors_}")
            actors_[actor.id] = actor

    actors = parse_geom_path(geom_path, actors_)
    parse_types_path(types_path, actors_)

    for activity in activities:
        for i in range(len(activity.actors)):
            actor: Actor = activity.actors[i]
            activity.actors[i] = actors[actor.id]

    return activities


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    yaml_path = Path(
        "data/meva/annotations/2018-03-07/16/2018-03-07.16-50-00.16-55-00.admin.G329.activities.yml"
    )
    parse_yaml(yaml_path)
