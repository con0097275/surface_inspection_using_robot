from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

import os
import seaborn as sns
import random
import matplotlib.pyplot as plt
import requests
import base64

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)  # if you have second camera you can set first parameter as 1
camera.set(cv2.CAP_PROP_FRAME_WIDTH , 320) # you should chose a value that the camera supports
camera.set(cv2.CAP_PROP_FRAME_HEIGHT , 240)



def detect_crack(image, contour_threshold=105, local_threshold=50, avg_thres=20):  ##edit
    # read image
    # image = cv2.imread(image_name)
    count = 0
    avg = 0
    thres = []
    for i in range(4):
        retval = cv2.getGaborKernel(ksize=(5, 5), sigma=10, theta=45 * i, lambd=5, gamma=1)
        gabor = cv2.filter2D(image, -1, retval)
        gray = cv2.cvtColor(gabor, cv2.COLOR_BGR2GRAY)
        (T, threshInv_) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        thres.append(threshInv_)

    threshInv_ = thres[0] | thres[1] | thres[2] | thres[3]
    contours_, h = cv2.findContours(threshInv_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours_:
        data = [len(x) for x in contours_]
        data = np.array(data)
        # count = data.max()
        # avg = data.mean()
        #### edit
        data = data[data > avg_thres]
        if (len(data) == 0):
            count = 0
        else:
            count = data.max()

    if (count >= contour_threshold):
        return True
    return False

def image_detector():
# while True:
    # Grab the webcamera's image.
    ret, image = camera.read()
    res, frame=   cv2.imencode('.jpg',image) 
    data = base64.b64encode(frame).decode("utf-8")
    image = cv2.resize(image, (448,448), interpolation=cv2.INTER_AREA)

    
    if (detect_crack(image)):
        api_post= {
            "image" : data
        }
        url= "https://fault-anomaly-detection-api-k6hgw7qjeq-ue.a.run.app/predict"
        response = requests.post(url, json = api_post).json()
        if len(response['image'])>102400:
            print("image too large")
            with open("image/bigimg.png", "rb") as img_file_t:
                temp = base64.b64encode(img_file_t.read())
            bigsize_image= temp.decode("utf-8")
            return (response['type'],bigsize_image)
        else:
            return (response['type'],response['image'])
    else:
        return ("Normal",data)

# camera.release()
# cv2.destroyAllWindows()