from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import cv2
import json
from prata.common.coco import get_imageid_to_ann, get_imageid_to_fname, get_label_id_from_categories
from prata.common.boxes import ious, iou, is_in_box
from decord import VideoReader, cpu

def crop_for_one_video(video_path: Path, annotation_path: Path, output_path:Path, num_negs: int):

    with open(annotation_path, "r") as f:
        obj = json.load(f)
    total_frames = len(obj["images"])
    pos_fids = set(map(lambda x: x["image_id"], obj["annotations"]))
    pos_roi =  obj["annotations"][0]["bbox"]
    pos_ltwh = np.array(pos_roi, dtype=np.int32)
    w,h = pos_ltwh[2:]
    offset = 0
    if (w!=h):
        offset = abs(w-h)
        idx = 0 if w < h else 1
        pos_ltwh[idx] -= offset // 2
        pos_ltwh[2+idx] = max(w,h)
    assert pos_ltwh[2] == pos_ltwh[3], f"Box hasn't squared: {pos_ltwh}"
   
    pos_ltbr = pos_ltwh.copy()
    pos_ltbr[2:] += pos_ltbr[ 0:2]
    x1,y1,x2,y2 = pos_ltbr

    vr = VideoReader(video_path.as_posix(), ctx=cpu(0))
    pos_frames = vr.get_batch(list(pos_fids)).asnumpy()
    pos_cap = cv2.VideoWriter((output_path/"pos"/video_path.name).as_posix(),
                              cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                              10, 
                              (pos_ltwh[2],pos_ltwh[3]))
    for frame in pos_frames:
        cropped = frame[y1:y2,x1:x2]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        pos_cap.write(cropped)
    pos_cap.release()
    
    neg_start_frames = [random.randint(0, int(total_frames/4*3)) for _ in range(num_negs)]
    pbar = tqdm(neg_start_frames)
    pbar.set_description("Cutting negatives")
    for i, neg_start_frame in enumerate(pbar):
        max_length = min(10*60, total_frames - neg_start_frame) # max 1min == 600 frames
        min_length = min(50, max_length//2)
        vid_length = random.randint(min_length, max_length)
        neg_end_frame = neg_start_frame + vid_length

        need_check_overlap_roi = False
        if neg_end_frame > min(pos_fids):
            need_check_overlap_roi = True
        vid_size = random.randint(400,800)
        x = None 
        y = None
        neg_ltrb = None
        while True:
            x = random.randint(0,1920-vid_size-1)
            y = random.randint(0,1080-vid_size-1)
            neg_ltrb = np.array([x,y,x+vid_size,y+vid_size],dtype=np.int32)
            if need_check_overlap_roi:
                iou_ = iou(pos_ltbr, neg_ltrb)
                if iou_ < 0.2:
                    break
            else:
                break
        assert neg_ltrb is not None, neg_ltrb
        x1,y1,x2,y2 = neg_ltrb

        frames = vr.get_batch(list(range(neg_start_frame, neg_start_frame+vid_length))).asnumpy()
        neg_cap = cv2.VideoWriter((output_path/f"neg/{i}.mp4").as_posix(),
                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                                10, 
                                (vid_size,vid_size))
        for frame in frames:
            cropped = frame[y1:y2,x1:x2]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            neg_cap.write(cropped)
        neg_cap.release()


def create_rider_from_ann(annotation_path: Path, output_path: Path):
    image_path = annotation_path.parent.parent / "images"
    with open(annotation_path, "r") as f:
        obj = json.load(f)
    person_id, motorbike_id = get_label_id_from_categories(obj["categories"], ["person", "motorcycle"])
    imageid_to_ann = get_imageid_to_ann(obj)
    imageid_to_fname = get_imageid_to_fname(obj)
    for fid, anns in imageid_to_ann.items():
        ppl, bikes = [], []
        for ann in anns:
            if ann["category_id"] == person_id:
                ppl.append(ann)
            elif ann["category_id"] == motorbike_id:
                bikes.append(ann)
        if len(ppl) == 0 or len(bikes) == 0:
            continue

        ppl_ltwh = np.array(list(map(lambda x:x["bbox"], ppl)), dtype=np.int32)
        bks_ltwh = np.array(list(map(lambda x:x["bbox"], bikes)), dtype=np.int32)

        ppl_ltrb = ppl_ltwh.copy()
        bks_ltrb = bks_ltwh.copy()
        ppl_ltrb[:, 2:] += ppl_ltwh[:, :2]
        bks_ltrb[:, 2:] += bks_ltwh[:, :2]

        ppl_centers = ppl_ltwh[:, :2] + ppl_ltwh[:, 2:]/2
        bks_centers = bks_ltwh[:, :2] + bks_ltwh[:, 2:]/2

        for i, (bike, ltrb) in enumerate(zip(bikes, bks_ltrb)):
            ious_ = ious(np.array([ltrb]), ppl_ltrb)
            frame = imageid_to_fname[bike["image_id"]]
            frame_path = image_path / frame
            img = cv2.imread(frame_path.as_posix())
            # linked_idx, max_iou = np.argmax(ious_[0])
            counter = 0
            for j, iou_ in enumerate(ious_[0]):
                if iou_ > 0.3:
                    susp_ltwh = ppl_ltwh[j]
                    susp_ltrb = ppl_ltrb[j]
                    if not is_in_box(bks_centers[i], susp_ltrb):
                        continue
                    if susp_ltrb[3] > ltrb[3] + 33:
                        continue
                    dist = np.linalg.norm(ppl_centers[j] - bks_centers[i])
                    if dist > 49:
                        continue
                    counter += 1
                    x1,y1,x2,y2 = susp_ltrb
                    x1_,y1_,x2_,y2_ = ltrb
                    l = min(x1,x1_)
                    t = min(y1,y1_)
                    r = max(x2,x2_)
                    b = max(y2,y2_)

                    l = max(l-30,0)
                    t = max(t-30,0)
                    r = min(r+30, 1920)
                    b = min(b+30,1080)
                    cropped = img[t:b,l:r]
                    # cv2.rectangle(img, (susp_ltrb[:2]), (susp_ltrb[2:]), (0,0,255),2,2)
                    # font = cv2.FONT_HERSHEY_COMPLEX
                    # cv2.putText(img,f'{iou_:.3f}_{dist:.3f}',susp_ltrb[:2],font,2,(255,0,255),3)  #text,coordinate,font,size of text,color,thickness of font

                    # cv2.rectangle(img, (ltrb[:2]), (ltrb[2:]), (255,0,0),2,2)
                    # cv2.imwrite((output_path/f"{i}_{j}_{iou_:.3f}_{dist:.3f}_{dist_l1:.3f}.jpg").as_posix(), img)
            if counter > 0:
                (output_path/f"{counter}").mkdir(exist_ok=True, parents=True)
                cv2.imwrite((output_path/f"{counter}/{bike['image_id']}_{i}.jpg").as_posix(), cropped)
                

    