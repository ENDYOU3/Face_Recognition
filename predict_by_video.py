import os
import cv2
import pandas as pd

test_path = "./test_file/video"
casc_file = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
face_model = cv2.face.LBPHFaceRecognizer_create()
face_model.read('face_model.yml')
img_pixel = (64,64)
df = pd.read_csv("target.csv")
y_label = df.name

file_test = os.listdir(test_path)
video_path = os.path.join(test_path, file_test[-1])

def choose_input():
    while True:
        choice = input("Choose input : (0)Open Camera | (1)Open Video >> ")
        if choice == "0" or choice == "1": 
            return int(choice)
        else:
            print("Please enter only (0) or (1) !")

def detect_face(image, casc_file):
    frontal_face = cv2.CascadeClassifier(casc_file)
    bBoxes = frontal_face.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))
    return bBoxes

def resize_image(image, pixel=1250):
    dimensions = image.shape
    image_resize_factor = pixel/dimensions[1]
    image = cv2.resize(image, None, fx=image_resize_factor, fy=image_resize_factor)
    return image

if __name__ == "__main__":
    tools = {0:0, 1:video_path}
    choice = choose_input()
    file_name = tools[choice]
    cap = cv2.VideoCapture(file_name)
    print("Opening...")
    if cap.isOpened() == False:
        print("Can't open this file...")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            color_image = resize_image(frame)
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            bBoxes = detect_face(gray_image, casc_file)
            for bBox in bBoxes:
                [x, y, w, h] = bBox
                crop_face = gray_image[y:y+h, x:x+w]
                crop_face = cv2.resize(crop_face, img_pixel)
                # Predict
                [pred_label, pred_conf] = face_model.predict(crop_face)
                box_text = y_label[pred_label]
                cv2.rectangle(color_image, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(color_image, box_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
                cv2.imshow("Windows", color_image)

            if (cv2.waitKey(6) & 0xFF == 27):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()