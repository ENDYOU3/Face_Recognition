 # Crop face
import os
import cv2
import time

input_path = "train_file/face_samples/" 
casc_file = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
img_pixel = (64, 64)

def crop_face(image, bBox, file_name):
    [x, y, w, h] = bBox
    face = image[y:y+h, x:x+w]
    save_crop_image(image=face, file_name=file_name, size=img_pixel)

def save_crop_image(image, file_name, size):
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    if "crop_face" not in os.listdir('./train_file'):
        os.mkdir("crop_face")
    cv2.imwrite("train_file/crop_face/"+file_name, image)

if __name__ == "__main__":
    frontal_face = cv2.CascadeClassifier(casc_file)
    input_file = os.listdir("./"+input_path)
    print("Starting to detect faces and save the cropped images...")
    starting_time = time.time()
    for file_name in input_file:
        color_image = cv2.imread(input_path+file_name)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        bBoxes = frontal_face.detectMultiScale(image=gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))
        for idx, bBox in enumerate(bBoxes):
            crop_face(image=gray_image, bBox=bBox, file_name="{}_{}".format(idx, file_name))
    print("Finished {} images in {:.2f} Seconds." .format(len(input_file) , (time.time() - starting_time)))