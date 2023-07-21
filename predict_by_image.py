import cv2
import os 
import pandas as pd

case_file = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
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

def detect_face(image, case_file):
    frontal_face = cv2.CascadeClassifier(case_file)
    bBoxes = frontal_face.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))
    return bBoxes

def save_result(file, image):
    if "result" not in os.listdir():
        os.mkdir("result")
    result_path = "./result/"
    cv2.imwrite(os.path.join(result_path, file), image)
    

if __name__ == "__main__":
    test_path = "./test_file/"
    print("Initializing face predicting...")
    for image_file in os.listdir(test_path):
        print("\nfile : {}".format(image_file))
        path = os.path.join(test_path, image_file)
        color_image = cv2.imread(path)
        color_image = resize_image(color_image)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        bBoxes = detect_face(gray_image, case_file)
        for idx, bBox in enumerate(bBoxes):
            [x, y, w, h] = bBox
            crop_image = gray_image[y:y+h, x:x+w]
            crop_image = cv2.resize(crop_image, img_pixel)

            [pred_label, pred_conf] = face_model.predict(crop_image)
            box_text = y_label[pred_label]
            text = "[{}] {}".format(idx, box_text)
            font_color = (255, 0, 255)
            font_scale = 0.7

            print("Predicted person [{}]: {:8}".format(idx, y_label[pred_label]))

            cv2.rectangle(color_image, (x,y), (x+w,y+h), (0, 255, 0), 2)
            cv2.putText(color_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 2)

            save_result(image_file, color_image)
            
        cv2.imshow("Window", color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print("\nEnd processing...")