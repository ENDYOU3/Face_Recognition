import cv2
import os 
import pandas as pd

test_path = "./test_file/image"
casc_file = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
face_model = cv2.face.LBPHFaceRecognizer_create()
face_model.read('face_model.yml')
img_pixel = (64,64)
df = pd.read_csv("target.csv")
y_label = df.name

def resize_image(image, pixel=1250):
    dimensions = image.shape
    image_resize_factor = pixel/dimensions[1]
    image = cv2.resize(image, None, fx=image_resize_factor, fy=image_resize_factor)
    return image

def detect_face(image, casc_file):
    frontal_face = cv2.CascadeClassifier(casc_file)
    bBoxes = frontal_face.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))
    return bBoxes

def save_result(file, image):
    if "result" not in os.listdir():
        os.mkdir("result")
    result_path = "./result/"
    cv2.imwrite(os.path.join(result_path, file), image)

if __name__ == "__main__":
    print("Initializing face predicting...")
    for image_file in os.listdir(test_path):
        print("\nfile : {}".format(image_file))
        image_path = os.path.join(test_path, image_file)
        color_image = cv2.imread(image_path)
        color_image = resize_image(color_image)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        bBoxes = detect_face(gray_image, casc_file)
        for bBox in bBoxes:
            [x, y, w, h] = bBox
            crop_image = gray_image[y:y+h, x:x+w]
            print(crop_image.shape)
            crop_image = cv2.resize(crop_image, img_pixel)
            print(crop_image.shape)
            # Predict
            [pred_label, pred_conf] = face_model.predict(crop_image)
            box_text = y_label[pred_label]
            print("Predicted person: {:8}".format(y_label[pred_label]))
            cv2.rectangle(color_image, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(color_image, box_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

        save_result(image_file, color_image)
        cv2.imshow("Window", color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print("\nEnd processing...")