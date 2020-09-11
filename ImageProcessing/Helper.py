import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class Helper():

    def __init__(self):
        '''self.dist = dist
        self.resize_percentage = resize_percentage'''

    ####  -*-*-*-*-*-*-**--*-*-*-*-*-*-*   #####

    def pre_processing(self, color_img):
        gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

        gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)

        binary_image = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                            cv.THRESH_BINARY, 11, 2)

        # Contrast adjusting with gamma correction y = 1.2
        # gray_img = np.array(255 * (gray_img / 255) ** 3, dtype='uint8')

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
        morph = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)
        morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel)
        morph = cv.morphologyEx(morph, cv.MORPH_DILATE, kernel)

        morph = cv.medianBlur(morph, 7)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        morph = cv.morphologyEx(morph, cv.MORPH_DILATE, kernel)

        final = cv.medianBlur(morph, 7)

        final, contours, hierarchy = cv.findContours(final, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        contours = np.delete(contours, 0, 0)

        sperm_size = 0
        for contour in contours:
            if 40 < np.size(contour) < 110:
                cv.polylines(color_img, contour, True, (0, 0, 255), 2)
                sperm_size = sperm_size + 1
            else:
                cv.polylines(color_img, contour, True, (255, 255, 0), 1)

                # cv.drawContours(color_img, contours, -1, (200, 32, 255), 2)

                # print(np.size(contour))

        return color_img, sperm_size

    ####  -*-*-*-*-*-*-**--*-*-*-*-*-*-*   #####

    def prepare_frame(self, img):
        '''kernel = np.ones((1, 1), np.uint8)
        erosion1 = cv2.erode(img, kernel, iterations=4)'''

        # l_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        # final = cv2.filter2D(final, -1, l_kernel)

        # (thresh, final) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        final = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                     cv.THRESH_BINARY, 11, 2)

        final = cv.medianBlur(final, 3)
        final = cv.medianBlur(final, 7)

        # final = cv2.GaussianBlur(final, (3,3), 0)

        cv.imshow("output", final)
        cv.imwrite('output.jpg', final)

        cv.waitKey()
        cv.destroyAllWindows()

    def thresholding(self, img):
        img = cv.GaussianBlur(img, (5, 5), 0)

        binary_image = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                            cv.THRESH_BINARY, 11, 2)

        return binary_image

    '''def noise2(self, img_rgb):
        # load color image
        # img_rgb = cv2.imread('input.jpg')

        # smooth the image with alternative closing and opening
        # with an enlarging kernel
        morph = img_rgb.copy()

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
        morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)
        morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))

        # take morphological gradient
        gradient_image = cv.morphologyEx(morph, cv.MORPH_GRADIENT, kernel)

        # split the gradient image into channels
        image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

        channel_height, channel_width, _ = image_channels[0].shape

        # apply Otsu threshold to each channel
        for i in range(0, 3):
            _, image_channels[i] = cv.threshold(~image_channels[i], 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
            image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

        # merge the channels
        image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)

        # save the denoised image
        cv.imshow("output", image_channels)
        cv.imwrite('output.jpg', image_channels)

        cv.waitKey()
        cv.destroyAllWindows()'''
