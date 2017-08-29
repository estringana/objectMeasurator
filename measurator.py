# import the necessary packages
import os

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# Variables to change
width_reference = 5.1

# Constants
minimum_area_to_be_detected_as_object = 30000
show_images = True


def extractBackground(imagePath):
    image = cv2.imread(imagePath)
    lower = np.array([150, 150, 150], dtype=np.uint8)
    upper = np.array([255, 255, 255], dtype=np.uint8)

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    # show the images
    showImage(output)

    return output

def showImage(image):
    if show_images:
        imS = cv2.resize(image, (1200, 800))  # Resize image
        cv2.imshow("output", imS)  # Show image
        cv2.waitKey(0)


# This function calculates the mid point between two coordinates
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def measureImage(imagePath):
    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(imagePath)
    showImage(image)
    # image = extractBackground(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    showImage(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    showImage(gray)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 20, 50)
    showImage(edged)
    edged = cv2.dilate(edged, None, iterations=1)
    showImage(edged)
    # edged = cv2.erode(edged, None, iterations=1)
    # showImage(edged)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    orig = image.copy()
    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < minimum_area_to_be_detected_as_object:
            continue

        print (cv2.contourArea(c))

        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / width_reference

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # draw the object sizes on the image
        cv2.putText(orig, "{:.1f}cm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}cm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

    showImage(orig)
    return orig


def detectBarcode(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show_images:
        cv2.imshow("Image", cv2.resize(gray, (1600, 900)))
        cv2.waitKey(0)

    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    if show_images:
        cv2.imshow("Image", cv2.resize(gradient, (1600, 900)))
        cv2.waitKey(0)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    if show_images:
        cv2.imshow("Image", cv2.resize(blurred, (1600, 900)))
        cv2.waitKey(0)
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    if show_images:
        cv2.imshow("Image", cv2.resize(closed, (1600, 900)))
        cv2.waitKey(0)
    closed = cv2.dilate(closed, None, iterations=4)
    if show_images:
        cv2.imshow("Closed", cv2.resize(closed, (1600, 900)))
        cv2.waitKey(0)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    for c in cnts:
        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    # draw a bounding box arounded the detected barcode and display the
    # image

    cv2.imshow("Final", cv2.resize(image, (1600, 900)))
    cv2.waitKey(0)


for filename in os.listdir('input'):
    extension = filename.lower()
    if extension.endswith(".jpg") or extension.endswith(".png"):
        print ('Image ' + filename)
        print ('    Measuring')
        result = measureImage('input/' + filename)
        print ('    Done')
        print ('    Reading barcodes')
        # detectBarcode('input/' + filename)
        print ('    Done')
        cv2.imwrite("output/" + filename, result)
        # imS = cv2.resize(result, (1600, 900))  # Resize image
        # cv2.imshow("output", imS)  # Show image
        # cv2.waitKey(0)

cv2.destroyAllWindows()
