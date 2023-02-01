from pathlib import Path
import random
import numpy as np
import shutil
from tqdm import tqdm
import cv2

list_categories = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
list_categories = list(map(lambda x: x.lower().strip(), list_categories))