# Face Recognition [ Python used OpenCV ]
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
   -  Collect samples of video file in the `test_file` folder.
      -  `test_file` >> `image` for samples image
      -  `test_file` >> `video` for samples video file
### Train model
```bash
   python training.py
```
   -  The result will be a face image. is stored in the `crop_face` folder. 
   -  Move and divide into groups from the `crop_face` folder to the `group_face` folder.

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



## Credit
   -  Image : https://kpopping.com/
