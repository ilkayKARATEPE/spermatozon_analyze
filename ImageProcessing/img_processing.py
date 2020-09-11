import os
import cv2 as cv
import numpy as np

import scipy
from scipy import ndimage

import Helper

helper_class = Helper.Helper()

# color_img = cv.imread("frame.jpg")
# gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

# helper_class.noise(cv2.imread("frame.jpg"))

# helper_class.prepare_frame(gray_img)

# helper_class.thresholding(gray_img)

# helper_class.noise(helper_class.thresholding(gray_img))


####  -*-*-*-*-*-*-*--*-*-*-*-*-*-*-*--*   ####

video_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + os.sep + 'video_sperm.mp4'

cap = cv.VideoCapture(video_path)

fourcc = cv.VideoWriter_fourcc(*'MP4V')
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (1280, 1024))

while True:
    ret, img = cap.read()

    img = helper_class.pre_processing(img)[0]

    position = (500, 40)
    cv.putText(
        img,
        "Spermatozan count: " + str(helper_class.pre_processing(img)[1]),  # text
        position,
        cv.FONT_HERSHEY_SIMPLEX,  # font family
        1,  # font size
        (50, 40, 50, 50),  # font color
        3)  # font stroke
    out.write(img)
    cv.imshow("Spermatozanlar painting ", img)
    if cv.waitKey(30) & 0xff == 27:
        break

cap.release()
out.release()
cv.destroyAllWindows()



# helper_class.noise(gray_correct)


cv.waitKey()
cv.destroyAllWindows()
