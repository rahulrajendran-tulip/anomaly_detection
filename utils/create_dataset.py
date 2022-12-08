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
    "/Users/rahul.rajendran/Desktop/Pytorch/anomaly_detection/datasets/visa_finetune"
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

train_folder = os.path.join(save_folder, "train")
test_folder = os.path.join(save_folder, "test")
mask_folder = os.path.join(save_folder, "mask")
train_mask_folder = os.path.join(mask_folder, "train")
test_mask_folder = os.path.join(mask_folder, "test")

# create normal and anomaly folders
train_img_normal_folder = os.path.join(train_folder, "normal")
train_img_anomaly_folder = os.path.join(train_folder, "anomaly")
test_img_normal_folder = os.path.join(test_folder, "normal")
test_img_anomaly_folder = os.path.join(test_folder, "anomaly")
train_mask_anomaly_folder = os.path.join(train_mask_folder, "anomaly")
test_mask_anomaly_folder = os.path.join(test_mask_folder, "anomaly")

# _mkdirs_for_dataset(train_img_normal_folder)
# _mkdirs_for_dataset(train_img_anomaly_folder)
# _mkdirs_for_dataset(test_img_normal_folder)
# _mkdirs_for_dataset(test_img_anomaly_folder)
# _mkdirs_for_dataset(train_mask_anomaly_folder)
# _mkdirs_for_dataset(test_mask_anomaly_folder)

with open(split_file, "r") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        object, set, label, image_path, mask_path = row
        if label == "normal":
            label = "normal"
        else:
            label = "anomaly"
        image_name = image_path.split("/")[-1]
        mask_name = mask_path.split("/")[-1]
        img_src_path = os.path.join(data_folder, image_path)
        msk_src_path = os.path.join(data_folder, mask_path)
        img_dst_path = os.path.join(save_folder, set, label, image_name)
        msk_dst_path = os.path.join(save_folder, "mask", set, label, mask_name)
        shutil.copyfile(img_src_path, img_dst_path)

        if label == "anomaly":
            mask = Image.open(msk_src_path)

            # binarize mask
            mask_array = np.array(mask)
            mask_array[mask_array != 0] = 255
            mask = Image.fromarray(mask_array)

            mask.save(msk_dst_path)
