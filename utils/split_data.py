# organize dataset into a useful structure
from os import listdir, makedirs
from random import random, seed
from shutil import copyfile

# create directories
dataset_home = "datasets/"
subdirs = ["train/", "test/"]
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ["normal/", "anomaly/"]
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)

# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = "datasets/all_images/"
for file in listdir(src_directory):
    src = src_directory + "/" + file
    dst_dir = "train/"
    if random() < val_ratio:
        dst_dir = "test/"
    if file.startswith("normal"):
        dst = dataset_home + dst_dir + "normal/" + file
        copyfile(src, dst)
    elif file.startswith("anomaly"):
        dst = dataset_home + dst_dir + "anomaly/" + file
        copyfile(src, dst)
