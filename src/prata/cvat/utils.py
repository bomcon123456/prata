import requests
from pprint import pprint
from natsort import natsorted
import time

server = "https://cvat-guardpro.vinai-systems.com"
api = "/api/v1"
tasks = "/tasks"
auth = ("gp-vinai", "thfGihj1IHFEZ1aU")

def get_all_task(project_id):
    req = f"{server}/api/v1/projects/{project_id}/tasks?page_size=10000"
    response = requests.get(req, auth=auth)
    task_list = response.json()["results"]
    ids = [x["id"] for x in task_list]
    names = [x["name"] for x in task_list]
    return ids, names

def create_tasks(project_id, list_data, annotation_dir=None, format="COCO%201.0", subset=None, task_name_with_parent=False):
    """
        Given folder of exported data, create tasks with name of zips with annotations
        project_id: project id to create task
        data_dir: folder consisiting exported dataset
        annotation_dir: folder consisting dumped annotation, if not given, use the annotation in the exported dataset
        format: format of the dumped annotation (COCO or CVAT)
    """
    for datapath in natsorted(list_data):
        ext = datapath.suffix.lower()
        if ext != ".zip":
            continue
        filename = datapath.stem
        print(filename)
        print("Creating task:", filename)

        task_name = filename if task_name_with_parent is False else f"{datapath.parent.name}_{filename}"
        data = {
            "name": task_name,
            "image_quality": 100,
            "project_id": project_id,
            "mode": "annotation",
            "overlap": 0,
            "segment_size": 0,
            "status": "annotation",
        }
        if subset is not None:
            data["subset"] = subset

        req = server + api + tasks
        print("Send POST API to create task...")
        req = f"{server}/api/v1/tasks?page=1"
        response = requests.post(req, json=data, auth=auth)
        # print("Create Task response:", response.json())

        task_id = response.json()["id"]
        req = server + api + tasks + f"/{task_id}/data"

        file = {
            "client_files[0]": open(datapath, "rb"),
            "Content-Type": "application/zip",
        }
        data = {"image_quality": 100}

        print("Send POST API to upload image to task...")
        response = requests.post(req, data=data, files=file, auth=auth)
        response = requests.post(req, data=data, files=file, auth=auth)
        print("")
        while True:
            response = requests.get(
                f"{server}/api/v1/tasks/{task_id}/status", auth=auth
            )
            time.sleep(0.5)
            print(
                "Upload images status:", response.status_code, response.json(), end="\r"
            )
            response = response.json()
            if response["state"] == "Finished":
                break

        if annotation_dir is not None:
            req2 = server + api + tasks + f"/{task_id}/annotations?format={format}"
            annotation_path = annotation_dir / filename / "annotations/instances_default.json"
            print("Anno path:", annotation_path)

            file2 = {
                "annotation_file": open(annotation_path, "rb"),
                # "Content-Type": "application/zip"
            }

            print("Send POST API to upload annotations to task...")
            print("")
            while True:
                response = requests.put(req2, files=file2, auth=auth)
                if response.status_code == 201:
                    print("Upload annations completed:", response.status_code)
                    break
                else:
                    print(
                        "Retry upload annotation:",
                        req2,
                        response.status_code,
                        response.text,
                        end="\r",
                    )
            # pprint(response.json())
            response = requests.get(f"{server}/api/v1/tasks/{task_id}/status", auth=auth)
            print("Upload annotations status:")
            print(response.status_code)
            pprint(response.json())
        print("Done task:", filename)
    # return response

def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def max_ioa(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    ioa_a = interArea/boxAArea
    ioa_b = interArea/boxBArea
    return max(ioa_a, ioa_b)


if __name__ == "__main__":
    format_cvat = "CVAT%201.1"
    format_coco = "COCO%201.0"
    data_path = "/home/ubuntu/workspace/datnh21/datalake/deliver47_v2/zip"
    json_path = "/home/ubuntu/workspace/datnh21/datalake/deliver47_v2"
    project_id = 57 # Check the project id in the portal web
    create_tasks(project_id, data_path, json_path, format_coco)