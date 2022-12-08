# source: https://github.com/amazon-science/spot-diff/blob/main/utils/prepare_data.py
import csv
import os
import shutil

import numpy as np
from PIL import Image


def _mkdirs_for_dataset(path):
    if not os.path.exists(path):
        os.makedirs(path)


data_folder = "/Users/rahul.rajendran/Desktop/datasets/VisA_20220922"
save_folder = (
    "/Users/rahul.rajendran/Desktop/Pytorch/anomaly_detection/datasets/all_images"
)
split_file = (
    "/Users/rahul.rajendran/Desktop/datasets/VisA_20220922/split_csv/2cls_fewshot.csv"
)

# data_list = [
#     "candle",
#     "capsules",
#     "cashew",
#     "chewinggum",
#     "fryum",
#     "macaroni1",
#     "macaroni2",
#     "pcb1",
#     "pcb2",
#     "pcb3",
#     "pcb4",
#     "pipe_fryum",
# ]

with open(split_file, "r") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        object, set, label, image_path, mask_path = row

        image_name = label + "." + object + "." + image_path.split("/")[-1]
        img_src_path = os.path.join(data_folder, image_path)
        img_dst_path = os.path.join(save_folder, image_name)
        shutil.copyfile(img_src_path, img_dst_path)
