## Glam Hack Preprocess

#Skeleton-recognition, calculating points of interest and saving this data

import pose_extract as pe
import matplotlib as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
import cv2
import os
import pymongo

CLIENT = pymongo.MongoClient("mongodb+srv://admin:QJ1KYlgo3kP8AYCk@cluster0.b6tbfym.mongodb.net/test")
DB = CLIENT["poseBuddy"]
COL = DB["metadata"]
COL2 = DB["metadata_with_array"]

# Push to mongo database for O(logn) search
def push_to_mongo(object, col):
    col.insert(object)

# Get images and filenames from folder
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Extract metadata and call push_to_mongo
def handle_metadata(images, filenames):
    for i in tqdm(range(len(images))):
        image_objects = {}
        _, image_metadata = pe.extract(images[i])
        if (image_metadata == None):
            continue
        image_objects["Filename"] = filenames[i]
        image_objects["Joints"] = image_metadata
        image_objects["Vectors"] = pe.generate_vectors(image_metadata)
        shutil.copy("./raw_images/"+filenames[i], "./processed_images/"+filenames[i])
        push_to_mongo(image_objects, COL2)    


images, filenames = load_images_from_folder("./raw_images/")
handle_metadata(images, filenames)


