from collections import defaultdict
import numpy as np
from pydantic import BaseModel
from typing import Optional, List, Any, Union, Dict


class VideoInfo(BaseModel):
    src: str
    fps: int
    frames: int
    duration: float
    height: int
    width: int


class ObjectAnnotationValue(BaseModel):
    highlight: bool
    instance: Union[int, str]
    score: Union[Optional[float], str]
    labelId: int
    color: str
    x: int
    y: int
    width: int
    height: int


class ActionAnnotationValue(BaseModel):
    start: float
    end: float
    action: int
    object: int
    color: str
    description: str


class ObjectLabelData(BaseModel):
    id: int
    name: str
    color: str


class ActionLabelData(BaseModel):
    id: int
    name: str
    color: str
    objects: List[int]


class VidatConfig(BaseModel):
    objectLabelData: List[ObjectLabelData]
    actionLabelData: List[ActionLabelData]

    @property
    def action_names(self):
        return list(map(lambda x: x.name, self.actionLabelData))

    def get_action_name_from_id(self, id):
        for actionlabel in self.actionLabelData:
            if actionlabel.id == id:
                return actionlabel.name
        raise Exception(f"Invalid action id: {id}")


class VidatAnnotation(BaseModel):
    video: VideoInfo
    keyframeList: List
    objectAnnotationListMap: Dict[int, List[ObjectAnnotationValue]]
    actionAnnotationList: List[ActionAnnotationValue]


class VidatAnnotationFull(BaseModel):
    version: str
    annotation: VidatAnnotation
    config: VidatConfig

    def lookup_action_from_id(self, action_id):
        return self.config.get_action_name_from_id(action_id)

    def get_ltrb_from_action(self, action_id, start_frame, end_frame):
        relevant_annotations = []
        for frame_id in range(start_frame, end_frame):
            if frame_id not in self.annotation.objectAnnotationListMap:
                continue
            for annotation in self.annotation.objectAnnotationListMap[frame_id]:
                annotation: ObjectAnnotationValue
                if annotation.labelId == action_id:
                    relevant_annotations.append(annotation)
        instance_to_annotation = defaultdict(list)
        for annotation in relevant_annotations:
            instance_to_annotation[annotation.instance].append(annotation)
        result = []
        for instance_id, annotations in instance_to_annotation.items():
            # for i in range(1, len(annotations)):
            # assert (
            #     annotations[0].x == annotations[i].x
            #     and annotations[0].y == annotations[i].y
            #     and annotations[0].width == annotations[i].width
            #     and annotations[0].height == annotations[i].height
            # ), f"{annotations[0]} vs {annotations[i]}"
            result.append(
                np.array(
                    [
                        annotations[0].x,
                        annotations[0].y,
                        annotations[0].x + annotations[0].width,
                        annotations[0].y + annotations[0].height,
                    ]
                )
            )
        return result
