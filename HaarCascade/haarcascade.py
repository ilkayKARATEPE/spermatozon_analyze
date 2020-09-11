import os
import cv2
import numpy as np

spermotoza_ = cv2.CascadeClassifier('cascade.xml')

print(os.path.normpath(os.getcwd() + os.sep + os.pardir) + os.sep + 'video_sperm.mp4')

cap = cv2.VideoCapture(os.path.normpath(os.getcwd() + os.sep + os.pardir) + os.sep + 'video_sperm.mp4')

while True:
    ret, img = cap.read()
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    l_kernel = np.array([[0, 1, 0], [1, -6, 1], [0, 1, 0]])

    #gray = cv2.filter2D(gray, -1, l_kernel)

    faces = spermotoza_.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 20, 40), 2)

    cv2.imshow('video', img)
    if cv2.waitKey(30) & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()
