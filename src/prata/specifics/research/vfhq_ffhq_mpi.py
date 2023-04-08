import os, sys, glob
import numpy as np
import math
from tqdm import tqdm
import cv2
import face_alignment
from skimage import io
import argparse
import PIL.Image
from multiprocessing import Pool, cpu_count
from math import ceil
import face_align


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--name',
                        dest='name', help='name', type=str)
    parser.add_argument('--source_text',
                        dest='source_text', help='Clips Annotation File', type=str)
    parser.add_argument('--source_folder',
                        dest='source_folder', help='Image folder', type=str)
    parser.add_argument('--type',
                        dest='type', help='Train/Test', type=str)
    parser.add_argument('--save_folder',
                        dest='save_folder', help='Save angles to folder', type=str)
    parser.add_argument('--num_landmarks',
                        dest='num_landmarks', help='5/68', type=int)
    parser.add_argument('--size',
                        dest='size', help='512/1024', type=int)
    parser.add_argument('--verbose', action='store_true',
                        dest='verbose')
    args = parser.parse_args()
    return args


def chunk_into_n(lst, n):
    size = ceil(len(lst) / n)
    return list(
        map(lambda x: lst[x * size:x * size + size], list(range(n)))
    )


# def processing(arg):
#     list_id, type_folder, source_folder, num_landmarks, size, original_save_folder, save_folder, name = arg
def processing(list_id, type_folder, source_folder, num_landmarks, size, original_save_folder, save_folder, name):
    # list_id, type_folder, source_folder, num_landmarks, size, original_save_folder, save_folder, name = arg
    if num_landmarks == 68:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    pbar = tqdm(list_id)
    for annot_path in pbar:
        annot_path = annot_path.strip()
        with open(os.path.join(f"{type_folder}/txt/", f"{annot_path}.txt"), "r") as f:
            annot_file = f.readlines()
            f.close()
        
        annot_file = [i.strip() for i in annot_file]
        for line in annot_file:
            if line.startswith('Video'):
                clip_video_txt = line
            if line.startswith('H'):
                clip_height_txt = line
            if line.startswith('W'):
                clip_width_txt = line
            if line.startswith('FPS'):
                clip_fps_txt = line
            # get the coordinates of face
            if line.startswith('CROP'):
                clip_crop_bbox = line

        _, video_id, pid, clip_idx, frame_rlt = annot_path.split('+')
        
        frame_index_dict = dict()
        for i in tqdm(list(range(7, len(annot_file)-1))):
            line = annot_file[i].split()
            landmarks2d = line[6:]
            lm_5 = []
            i = 0
            for landmark in landmarks2d:
                if i % 2 == 0:
                    temp = [float(landmark.strip())]
                else:
                    temp.append(float(landmark.strip()))
                    lm_5.append(temp)
                i += 1
            lm_5 = np.array(lm_5, dtype=np.float32)
        
            lm_68 = None
            if num_landmarks == 68:
                image_path = os.path.join(source_folder, type_folder, video_id, annot_path, f"{line[0]}.png")
                if os.path.exists(image_path):
                    try:
                        input_img = io.imread(image_path)
                        if input_img.shape[2] == 4:
                            input_img = input_img[:, :, :3]
                        preds = fa.get_landmarks(input_img)
                        lm_68 = preds[0]
                    except Exception as e:
                        print(e)
                        pass

            frame_index_dict[line[0]] = {
                "annotation": annot_file[i],
                "bbox": [math.floor(float(line[2])), math.floor(float(line[3])), math.ceil(float(line[2])+float(line[4])), math.ceil(float(line[3])+float(line[5]))],
                "landmark_5": lm_5,
                "landmark_68": lm_68
            }

        # Pose Estimation
        for frame_index, values in tqdm(list(frame_index_dict.items())):
            try:
                # frame = int(frame_index)
                # bbox = values["bbox"]
                face_landmarks_5 = values["landmark_5"]
                face_landmarks_68 = values["landmark_68"]
                # Full Image
                image_path = os.path.join(source_folder, type_folder, video_id, annot_path, f"{frame_index}.png")

                if num_landmarks == 68:
                    if face_landmarks_68 is None:
                        face_landmarks = values["landmark_5"]
                    else:
                        face_landmarks = values["landmark_68"]

                if os.path.exists(image_path):
                    img = cv2.imread(image_path)
                    if face_landmarks.shape[0] == 5:
                        img, _, lm_trans_5 = face_align.image_align_5(img, face_landmarks, face_landmarks_5, output_size=size, transform_size=4096, enable_padding=True)
                    else:
                        img, _, lm_trans_5 = face_align.image_align_68(img, face_landmarks, face_landmarks_5, output_size=size, transform_size=4096, enable_padding=True)
                    if img is not None:
                        if not os.path.exists(os.path.join(save_folder, video_id, pid, clip_idx)):
                            os.makedirs(os.path.join(save_folder, video_id, pid, clip_idx), exist_ok=True)
                        cv2.imwrite(os.path.join(save_folder, video_id, pid, clip_idx, f"{frame_index}.png"), img)
                        for (x, y) in lm_trans_5:
                            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)
                        cv2.imwrite("test_lmk.png", img)
                        frame_index_dict[line[0]]["landmark_5"] = lm_trans_5
                else:
                    if os.path.exists(os.path.join(original_save_folder, "not_found", f"not_found_{name}.txt")):
                        not_found_file = open(os.path.join(original_save_folder, "not_found", f"not_found_{name}.txt"), "a")
                    else:
                        not_found_file = open(os.path.join(original_save_folder, "not_found", f"not_found_{name}.txt"), "w")
                    not_found_file.write(f"Not Exist Frames: {video_id}\t{frame_index}\n")
                    not_found_file.close()
            except Exception as e:
                print(e)
                if os.path.exists(os.path.join(original_save_folder, "not_found", f"not_found_{name}.txt")):
                    not_found_file = open(os.path.join(original_save_folder, "not_found", f"not_found_{name}.txt"), "a")
                else:
                    not_found_file = open(os.path.join(original_save_folder, "not_found", f"not_found_{name}.txt"), "w")
                not_found_file.write(f"{video_id}\t{frame_index}\t{e}\n")
                not_found_file.close()
                continue

        if not os.path.exists(f"{type_folder}/txt2"):
            os.makedirs(f"{type_folder}/txt2", exist_ok=True)
        with open(os.path.join(f"{type_folder}/txt2/", f"{annot_path}.txt"), "w") as f:
            f.write(f"{clip_video_txt}\n")
            f.write(f"{clip_height_txt}\n")
            f.write(f"{clip_width_txt}\n")
            f.write(f"{clip_fps_txt}\n")
            f.write(f"\n")
            f.write("FRAME INDEX X0 Y0 W H [Landmarks (5 Points)]\n")
            f.write(f"\n")
            for k, v in frame_index_dict.items():
                ann = v["annotation"].strip().split()
                new_lm2d = v["landmark_5"]
                new_list = [ann[0], ann[1], ann[2], ann[3], ann[4], ann[5]]

                line_new = " ".join(new_list)
                for row in new_lm2d:
                    coord_x = " {:.2f}".format(row[0])
                    coord_y = " {:.2f}".format(row[1])
                    line_new += f"{coord_x}{coord_y}"
                f.write(f"{line_new}\n")
            f.write(f"{clip_crop_bbox}\n")
            f.close()

        if os.path.exists(os.path.join(original_save_folder, "ckpt", f"ckpt_{name}.txt")):
            ckpt_file = open(os.path.join(original_save_folder, "ckpt", f"ckpt_{name}.txt"), "a")
        else:
            ckpt_file = open(os.path.join(original_save_folder, "ckpt", f"ckpt_{name}.txt"), "w")
        ckpt_file.write(f"{annot_path}\n")
        ckpt_file.close()



if __name__ == "__main__":
    args = parse_args()
    save_folder = args.save_folder
    source_folder = args.source_folder

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, "txt"), exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, "not_found"), exist_ok=True)

    save_folder = os.path.join(args.save_folder, args.type)
    os.makedirs(os.path.join(save_folder), exist_ok=True)
    
    with open(args.source_text, "r") as f:
        clip_file_list = f.readlines()
        f.close()
    
    print(len(clip_file_list))
    if os.path.exists(os.path.join(args.save_folder, "ckpt", f"ckpt_{args.name}.txt")):
        with open(os.path.join(args.save_folder, "ckpt", f"ckpt_{args.name}.txt"), "r") as f:
            done_id_list = f.readlines()
            f.close()
        clip_file_list = [i.strip() for i in clip_file_list if i not in done_id_list]
    print(len(clip_file_list))

    # processing(clip_file_list, args.type, args.source_folder, args.num_landmarks, args.size, args.save_folder, save_folder, args.name)
    
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = multiprocessing.cpu_count()
    
    chunked_list = chunk_into_n(clip_file_list, 5)
    new_process = []
    for i in chunked_list:
        new_arg = (i, args.type, args.source_folder, args.num_landmarks, args.size, args.save_folder, save_folder, args.name)
        new_process.append(new_arg)
    if args.verbose:
        print(new_process)
    with Pool(ncpus) as pool:
        pool.starmap(processing, new_process)
