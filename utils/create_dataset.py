# type: ignore
# source: https://github.com/amazon-science/spot-diff/blob/main/utils/prepare_data.py
import csv
import os
import shutil


def _mkdirs_for_dataset(path):
    if not os.path.exists(path):
        os.makedirs(path)


DATA_FOLDER = "/Users/rahul.rajendran/Desktop/datasets/VisA_20220922"
SAVE_FOLDER = (
    "/Users/rahul.rajendran/Desktop/Pytorch/anomaly_detection/datasets/all_images"
)
SPLIT_FILE = (
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

# pylint: disable=arguments-differ
with open(SPLIT_FILE, "r", encoding="utf-8") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        _, _, label, image_path, mask_path = row

        image_name = label + "." + object + "." + image_path.split("/")[-1]
        img_src_path = os.path.join(DATA_FOLDER, image_path)
        img_dst_path = os.path.join(SAVE_FOLDER, image_name)
        shutil.copyfile(img_src_path, img_dst_path)
