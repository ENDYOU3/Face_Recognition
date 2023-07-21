import os
import numpy as np
import cv2
import pandas as pd
import time

feature_path = "./group_face/"

def save_labels(target_name):
    df = pd.DataFrame(target_name, columns=['name'])
    df.to_csv("target.csv")
    print("Save labels...\n{}\n".format(target_name))

def get_data(path):
    target_id = 0
    images, labels = [], []
    target_name = []
    for subdir in os.listdir(path):
        for face_file in os.listdir(path+subdir):
            image = cv2.imread(path+os.path.sep+subdir+os.path.sep+face_file, cv2.IMREAD_GRAYSCALE)
            images.append(np.asarray(image, dtype=np.uint8))
            labels.append(target_id)
        target_id += 1
        target_name.append(subdir)

    return [images, labels, target_name]

def train_model(path):
    [images, labels, target_name] = get_data(path)
    labels = np.asarray(labels, dtype=np.int32)
    print("Total trained images: {}".format(len(images)))
    print("Total target: {}".format(len(target_name)))

    print("\nInitializing FaceRecognizer and training...")
    starting_time = time.time()
    face_model= cv2.face.LBPHFaceRecognizer_create()
    face_model.train(images, labels)
    print("Complete use {:.2f} seconds for training\n".format(time.time()-starting_time))
    return [face_model, target_name]

if __name__ in "__main__":
    face_model, target_name = train_model(feature_path)
    print("Model saving...\n")
    face_model.write("face_model.yml")
    save_labels(target_name)
    print("Training model completed!\n")