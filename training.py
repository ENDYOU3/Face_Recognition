import os
import numpy as np
import cv2


test_path = "./group_face/Lisa/Lisa (1).jpeg"
image = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, ())
# cv2.imshow("mypic", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()