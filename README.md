# Face Recognition [ Python used OpenCV ]
Face recognition is a method of identifying or verifying the identity of an individual using their face. Face recognition systems can be used to identify people in photos, video, or in real-time. 
## Feature
**Face detector**
   -  Find all the faces that appear in a picture
   ```python
      import cv2
      frontal_face = cv2.CascadeClassifier(casc_file)
      frontal_face.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))
   ```
**Face recognition**
   -  Identify faces in pictures
```python
   import cv2
   face_model= cv2.face.LBPHFaceRecognizer_create()
   face_model.train(images, labels)
```
## Installation

### Requirements :
**Programming Language :** `python`

**Libraries :** `pandas`,  `numpy`,  `opencv`

### Run with Integrated development environment (IDE)
**Create environment and install package :**
   -  Open file location on `terminal` or `cmd`
```bash
   cd "location path"
```
   -  install `pipenv`
```bash
   pip install pipenv
```
   -  Activate a virtual environment
```bash
   pipenv shell
```
   -  Install all packages in pipfile
```bash
   pipenv install
```


## Usage

### Data prepare
   -  Collect samples of images in the `face_samples` folder.
      - Collect only face images.
         ```bash
         python crop_face.py
         ```
         -  The result will be a face image. is stored in the `crop_face` folder. 
         -  Move and divide into groups from the `crop_face` folder to the `group_face` folder.
         
   -  Collect samples of video file in the `test_file` folder.
      -  `test_file` >> `image` for samples image
      -  `test_file` >> `video` for samples video file

### Train model
```bash
   python training.py
```
   -  use face in `group_face` folder.

### Predict model
**predicting model has 2 options**
   -  Predicting by image >> The result is stored in the `result` folder.
```bash
   python predict_by_image.py
```
   -  Predicting by video file or camera
```bash
   python predict_by_video.py
```

## Screenshots
![predict_by_image](https://github.com/ENDYOU3/Face_Recognition/assets/103243756/7191be42-8fae-4491-a342-2e9f1dd88946)


## Credit
   -  Image : https://kpopping.com/
